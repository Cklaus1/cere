"""Reference safetensors header parser, family classifier and model inventory (M2.1 Q5, Q12, Q15, Q16, Q17).

Reads only the 8+n header region of every shard plus the HF tree metadata; never tensor data.
`python3 scripts/model_inventory.py SRC --families F [--out FILE]` prints the canonical inventory JSON.
`python3 scripts/model_inventory.py --dump-header SHARD` prints one shard's parsed header (data-offset order).
Rejections exit 2 with `<error_class>: <message>` as the first stderr line.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import struct
import sys
from pathlib import Path

from cere_contracts import (DTYPE_ITEMSIZE, MAX_U64, ArtifactError, ContractError, canonical_json, family_totals,
                            json_int, limits_with, parse_json, validate)

_DIGITS = re.compile(r"[0-9]+", re.ASCII)
_EXPERT = re.compile(r"(?:^|\.)layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.", re.ASCII)
_PLE_SHARD = re.compile(r"shard_([0-9]+)\.weight", re.ASCII)
NVFP4_SUFFIX = {"nvfp4_weight": ".weight", "nvfp4_block_scale": ".weight_scale",
                "nvfp4_global_scale": ".weight_scale_2", "nvfp4_input_scale": ".input_scale"}
TREE_DIR = ".cache/huggingface/trees"


class FileIO:
    """Injectable file access; tests subclass it to count or fault reads (B3)."""

    def open(self, name, flags, dir_fd=None):
        return os.open(name, flags, dir_fd=dir_fd)

    def fstat(self, fd):
        return os.fstat(fd)

    def pread(self, fd, n, offset):
        out = bytearray()
        while len(out) < n:
            chunk = os.pread(fd, min(n - len(out), 8 << 20), offset + len(out))
            if not chunk:
                break
            out += chunk
        return bytes(out)

    def close(self, fd):
        os.close(fd)

    def listdir(self, fd):
        return os.listdir(fd)

    def lstat(self, name, dir_fd):
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)


FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def open_regular(io: FileIO, name: str, dir_fd: int) -> tuple[int, os.stat_result]:
    """openat(dir_fd, name, O_NOFOLLOW) + fstat S_ISREG. Symlink -> `symlink`, anything else -> `special_file`."""
    try:
        fd = io.open(name, FILE_FLAGS, dir_fd=dir_fd)
    except OSError as e:
        if e.errno == errno.ELOOP:  # final component is a symlink
            raise ArtifactError("symlink", f"{name} is a symlink") from None
        raise
    st = io.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        io.close(fd)
        raise ArtifactError("special_file", f"{name} is not a regular file")
    return fd, st


def _header_pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ArtifactError("duplicate_key", f"duplicate header key: {key}")
        out[key] = value
    return out


def _reject_constant(value):
    raise ArtifactError("header_json", f"non-JSON constant {value}")


def read_header_prefix(io: FileIO, fd: int, file_size: int, limits: dict) -> bytes:
    """Bytes [0, 8+n) of a shard, bounds-checked before the header is read (Q15)."""
    if file_size < 8:
        raise ArtifactError("header_bounds", "file shorter than the 8-byte header length")
    n = struct.unpack("<Q", io.pread(fd, 8, 0))[0]
    if n < 2 or n > limits["max_header_bytes"] or 8 + n > file_size:
        raise ArtifactError("header_bounds", f"header length {n} out of bounds for file size {file_size}")
    head = io.pread(fd, n, 8)
    if len(head) != n:
        raise ArtifactError("header_bounds", "short header read")
    return struct.pack("<Q", n) + head


def parse_header(prefix: bytes, file_size: int, limits: dict) -> dict:
    """Parse a bounds-checked header prefix. Returns metadata and tensors (data-offset order) with absolute
    offsets, byte sizes and element alignment. Raises ArtifactError with a header class (contract order)."""
    n = len(prefix) - 8
    try:
        text = prefix[8:].decode("utf-8")
    except UnicodeDecodeError:
        raise ArtifactError("header_utf8", "header is not UTF-8") from None
    try:
        obj = json.loads(text, object_pairs_hook=_header_pairs, parse_constant=_reject_constant, parse_int=json_int)
    except ArtifactError:
        raise
    except (ValueError, RecursionError) as e:
        raise ArtifactError("header_json", f"invalid header JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ArtifactError("header_json", "header is not a JSON object")
    metadata = obj.get("__metadata__")
    if "__metadata__" in obj and (not isinstance(metadata, dict) or
                                  not all(isinstance(v, str) for v in metadata.values())):
        raise ArtifactError("metadata", "__metadata__ must be a string to string map")
    entries = [(k, v) for k, v in obj.items() if k != "__metadata__"]
    for name, e in entries:
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            raise ArtifactError("tensor_fields", f"{name}: fields must be exactly dtype, shape, data_offsets")
        off = e["data_offsets"]
        if (not isinstance(off, list) or len(off) != 2 or
                any(type(x) is not int or x < 0 or x > MAX_U64 for x in off)):
            raise ArtifactError("tensor_fields", f"{name}: data_offsets must be two unsigned integers")
        if not isinstance(e["dtype"], str) or e["dtype"] not in DTYPE_ITEMSIZE:
            raise ArtifactError("dtype", f"{name}: unknown dtype {e['dtype']!r}")
        shape = e["shape"]
        if (not isinstance(shape, list) or len(shape) > 16 or
                any(type(d) is not int or d < 1 or d > MAX_U64 for d in shape)):
            raise ArtifactError("shape", f"{name}: shape must be 0..16 dims, each >= 1")
    if len(entries) > limits["max_tensors_per_file"]:
        raise ArtifactError("too_many_tensors", f"{len(entries)} tensors > {limits['max_tensors_per_file']}")
    base = 8 + n
    data_len = file_size - base
    tensors = []
    for name, e in entries:
        begin, end = e["data_offsets"]
        if end < begin:
            raise ArtifactError("offsets", f"{name}: end < begin")
        size = DTYPE_ITEMSIZE[e["dtype"]]
        for d in e["shape"]:
            size *= d
            if size > MAX_U64:
                raise ArtifactError("byte_size", f"{name}: prod(shape) x itemsize overflows u64")
        if size != end - begin:
            raise ArtifactError("byte_size", f"{name}: {size} bytes declared by shape, {end - begin} by offsets")
        tensors.append({"name": name, "dtype": e["dtype"], "shape": e["shape"], "begin": begin, "end": end})
    tensors.sort(key=lambda t: (t["begin"], t["end"]))
    pos = 0
    for t in tensors:
        if t["begin"] != pos:
            raise ArtifactError("offsets", f"{t['name']}: data region has a gap or overlap at {pos}")
        pos = t["end"]
    if pos != data_len:
        raise ArtifactError("offsets", f"data region ends at {pos}, file data is {data_len} bytes")
    out = []
    for t in tensors:
        offset = base + t["begin"]
        out.append({"name": t["name"], "dtype": t["dtype"], "shape": t["shape"], "offset": offset,
                    "bytes": t["end"] - t["begin"], "element_aligned": offset % DTYPE_ITEMSIZE[t["dtype"]] == 0})
    return {"header_bytes": base, "header_sha256": hashlib.sha256(prefix).hexdigest(), "metadata": metadata,
            "tensors": out}


class Families:
    """Compiled family rules (Q5): every tensor fullmatches exactly one rule."""

    def __init__(self, rules_doc: dict, placement: dict | None = None):
        self.doc = rules_doc
        self.rules = rules_doc["rules"]
        self.groups = rules_doc["groups"]
        self.compiled = [re.compile(r["pattern"], re.ASCII) for r in self.rules]
        # inventory: placement from the rule; manifest: from the embedded profile
        self.placement = [placement[r["role"]] if placement is not None else r["placement"] for r in self.rules]

    def classify(self, t: dict) -> int:
        hits = [i for i, c in enumerate(self.compiled) if c.fullmatch(t["name"])]
        if not hits:
            raise ArtifactError("family_unmatched", f"{t['name']} matches no family rule")
        if len(hits) > 1:
            raise ArtifactError("family_ambiguous",
                                f"{t['name']} matches {', '.join(self.rules[i]['family'] for i in hits)}")
        r = self.rules[hits[0]]
        if t["dtype"] != r["dtype"]:
            raise ArtifactError("family_dtype", f"{t['name']}: {t['dtype']} but family {r['family']} is {r['dtype']}")
        sr = r["shape_rule"]
        if sr != "any" and len(t["shape"]) != int(sr.split(":")[1]):
            raise ArtifactError("family_shape", f"{t['name']}: shape {t['shape']} violates {sr}")
        return hits[0]

    def tally(self, classified) -> list[dict]:
        """`classified`: iterable of (rule index, tensor)."""
        fams = [{"family": r["family"], "component": r["component"], "role": r["role"], "dtype": r["dtype"],
                 "placement": self.placement[i], "count": 0, "bytes": 0, "misaligned": 0}
                for i, r in enumerate(self.rules)]
        for i, t in classified:
            f = fams[i]
            f["count"] += 1
            f["bytes"] += t["bytes"]
            f["misaligned"] += 0 if t["element_aligned"] else 1
        return fams

    def check_groups(self, classified) -> None:
        """Group checks (Q5 a-c) over (rule index, tensor) pairs; order nvfp4, ple_table, ple_module."""
        by_role: dict = {}
        for i, t in classified:
            by_role.setdefault(self.rules[i]["role"], []).append(t)
        G = self.groups["nvfp4_group_size"]
        quads: dict = {}
        for role, suffix in NVFP4_SUFFIX.items():
            for t in by_role.get(role, []):
                if not t["name"].endswith(suffix):
                    raise ArtifactError("group_nvfp4", f"{t['name']}: role {role} needs suffix {suffix}")
                q = quads.setdefault(t["name"][:-len(suffix)], {})
                if role in q:
                    raise ArtifactError("group_nvfp4", f"{t['name']}: duplicate quad member")
                q[role] = t
        for prefix, q in sorted(quads.items()):
            if set(q) != set(NVFP4_SUFFIX):
                raise ArtifactError("group_nvfp4", f"{prefix}: incomplete NVFP4 quad {sorted(q)}")
            w, s = q["nvfp4_weight"]["shape"], q["nvfp4_block_scale"]["shape"]
            if len(w) != 2 or len(s) != 2:
                raise ArtifactError("group_nvfp4", f"{prefix}: weight and block scale must be rank 2")
            i = 2 * w[1]
            if i % G or s != [w[0], i // G]:
                raise ArtifactError("group_nvfp4", f"{prefix}: block scale {s} does not match weight {w}")
            if q["nvfp4_global_scale"]["shape"] != [] or q["nvfp4_input_scale"]["shape"] != []:
                raise ArtifactError("group_nvfp4", f"{prefix}: global/input scales must be scalars")
        table, scale, index = (by_role.get(r, []) for r in ("ple_table", "ple_scale", "ple_index"))
        if table or scale or index:
            ks = {}
            for t in table:
                m = re.search(r"(?:^|\.)" + _PLE_SHARD.pattern + r"\Z", t["name"], re.ASCII)
                if not m or int(m.group(1)) in ks:
                    raise ArtifactError("group_ple_table", f"{t['name']}: bad or duplicate PLE shard name")
                ks[int(m.group(1))] = t
            if set(ks) != set(range(self.groups["ple_table_shards"])):
                raise ArtifactError("group_ple_table",
                                    f"PLE shards {sorted(ks)} != 0..{self.groups['ple_table_shards'] - 1}")
            shapes = {tuple(t["shape"]) for t in table}
            if len(shapes) != 1 or len(next(iter(shapes))) != 2:
                raise ArtifactError("group_ple_table", f"PLE shard shapes differ or are not rank 2: {shapes}")
            if table and table[0]["dtype"] == "U8":
                self._check_ple_nvfp4_scales(next(iter(shapes)), set(ks), scale)
            elif len(scale) != 1 or scale[0]["shape"] != [1]:
                raise ArtifactError("group_ple_table", "PLE table needs exactly one weight_scale of shape [1]")
            _members(index, self.groups["ple_index_members"], "group_ple_table", "PLE index")
        module = by_role.get("ple_module", [])
        if module:
            _members(module, self.groups["ple_module_members"], "group_ple_module", "PLE module")
            if any(t["dtype"] != "BF16" for t in module):
                raise ArtifactError("group_ple_module", "PLE module weights must be BF16")

    def _check_ple_nvfp4_scales(self, table_shape, shards: set, scale) -> None:
        """D8 NVFP4 PLE (tasks/fp4local-spec.md): the U8 table [rows, cols] holds two E2M1 values per byte; each shard
        has an F8_E4M3 `shard_<k>.weight_scale` [rows, 2*cols/nvfp4_group_size], plus exactly one F32 [1]
        `weight_scale_2`. Mirrors families.rs `check_ple_nvfp4_scales`."""
        rows, cols = table_shape
        G = self.groups["nvfp4_group_size"]
        if G == 0 or (2 * cols) % G:
            raise ArtifactError("group_ple_table", "NVFP4 PLE table width is not a whole number of groups")
        want, seen, globals_ = [rows, 2 * cols // G], set(), 0
        for s in scale:
            m = re.search(r"(?:^|\.)shard_([0-9]+)\.weight_scale\Z", s["name"], re.ASCII)
            if m:
                k = int(m.group(1))
                if s["dtype"] != "F8_E4M3" or s["shape"] != want or k in seen:
                    raise ArtifactError("group_ple_table", f"{s['name']}: NVFP4 shard scale repeated or not "
                                                           f"F8_E4M3 {want}")
                seen.add(k)
            elif s["name"].endswith(".weight_scale_2") and s["dtype"] == "F32" and s["shape"] == [1]:
                globals_ += 1
            else:
                raise ArtifactError("group_ple_table", f"{s['name']}: neither an NVFP4 shard scale nor an F32 [1] "
                                                       "weight_scale_2")
        if seen != shards or globals_ != 1:
            raise ArtifactError("group_ple_table", "NVFP4 PLE needs one scale per shard and exactly one weight_scale_2")


def _members(tensors, members, cls, what) -> None:
    seen = {}
    for t in tensors:
        hit = [m for m in members if t["name"].endswith("." + m)]
        if len(hit) != 1 or hit[0] in seen:
            raise ArtifactError(cls, f"{t['name']}: not a (unique) {what} member")
        seen[hit[0]] = t
    if set(seen) != set(members):
        raise ArtifactError(cls, f"{what} members {sorted(seen)} != {sorted(members)}")


def check_expert_metadata(metadata, tensors) -> None:
    """Q5e: expert-shard `__metadata__` ranges agree with the tensor names."""
    keys = ("layer", "expert_start", "expert_end")
    if not metadata or not any(k in metadata for k in keys):
        return
    if not all(k in metadata and _DIGITS.fullmatch(metadata[k]) for k in keys):
        raise ArtifactError("expert_metadata", "layer/expert_start/expert_end must all be decimal strings")
    layer, start, end = (int(metadata[k]) for k in keys)
    seen = set()
    for t in tensors:
        m = _EXPERT.search(t["name"])
        if not m or int(m.group(1)) != layer or not start <= int(m.group(2)) < end:
            raise ArtifactError("expert_metadata", f"{t['name']}: outside layer {layer} experts [{start},{end})")
        seen.add(int(m.group(2)))
    if seen != set(range(start, end)):
        raise ArtifactError("expert_metadata", f"experts {sorted(seen)} != [{start},{end})")


def header_digest(shards: list[tuple[str, str]]) -> str:
    """Q12: sha256 over `<header sha256>  <name>\\n` lines ordered bytewise by file name (not by hash)."""
    lines = "".join(f"{sha}  {name}\n" for name, sha in sorted(shards, key=lambda s: s[0].encode()))
    return hashlib.sha256(lines.encode()).hexdigest()


def _read_small(io: FileIO, name: str, dir_fd: int, counter: dict) -> bytes:
    fd, st = open_regular(io, name, dir_fd)
    try:
        data = io.pread(fd, st.st_size, 0)
    finally:
        io.close(fd)
    counter["metadata_bytes_read"] += len(data)
    return data


def _source_tree(io: FileIO, src_fd: int, counter: dict):
    """The single HF tree file, or None when absent. More than one, or unparsable -> `source_tree`."""
    fd = None
    try:
        fd = src_fd
        for part in TREE_DIR.split("/"):
            nfd = io.open(part, DIR_FLAGS, dir_fd=fd)
            if fd != src_fd:
                io.close(fd)
            fd = nfd
    except FileNotFoundError:
        if fd not in (None, src_fd):
            io.close(fd)
        return None
    try:
        names = sorted(n for n in io.listdir(fd) if n.endswith(".json"))
        if len(names) > 1:
            raise ArtifactError("source_tree", f"{len(names)} HF tree files")
        if not names:
            return None
        raw = _read_small(io, names[0], fd, counter)
    finally:
        io.close(fd)
    return parse_tree(names[0], raw)


def _hex(value, n: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{n}}}", value) is not None


def parse_tree(name: str, raw: bytes) -> dict:
    """An HF tree file `<40-hex revision>.json` holding `{"files": {name: {size, blob_id, lfs_sha256?}}}` (sizes
    u64, blob ids 40 and lfs digests 64 lowercase hex, as the inventory and manifest schemas need); anything else is
    `source_tree`, as cerec's `source::parse_tree` (shared cases: tag `inventory` of artifact_invalid.json)."""
    stem = name[:-len(".json")] if name.endswith(".json") else ""
    try:
        tree = parse_json(raw)
        files = tree["files"]
        ok = (_hex(stem, 40) and isinstance(files, dict) and
              all(isinstance(e, dict) and type(e.get("size")) is int and 0 <= e["size"] <= MAX_U64 and
                  _hex(e.get("blob_id"), 40) and ("lfs_sha256" not in e or _hex(e["lfs_sha256"], 64))
                  for e in files.values()))
    except (ContractError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise ArtifactError("source_tree", f"unparsable HF tree file {name}")
    return {"revision": stem, "tree_sha256": hashlib.sha256(raw).hexdigest(), "files": files}


def list_shards(io: FileIO, src_fd: int) -> list[str]:
    """Top-level `*.safetensors`, bytewise order; a symlink or non-regular entry is rejected."""
    out = []
    for name in io.listdir(src_fd):
        if not name.endswith(".safetensors"):
            continue
        st = io.lstat(name, src_fd)
        if stat.S_ISLNK(st.st_mode):
            raise ArtifactError("symlink", f"{name} is a symlink")
        if not stat.S_ISREG(st.st_mode):
            raise ArtifactError("special_file", f"{name} is not a regular file")
        out.append(name)
    return sorted(out, key=str.encode)


def inventory(src: Path, families_path: Path, *, limits: dict | None = None, io: FileIO | None = None) -> dict:
    io = io or FileIO()
    lim = limits_with(limits)
    counter = {"metadata_bytes_read": 0}
    fam_raw = Path(families_path).read_bytes()
    fam_doc = parse_json(fam_raw)
    validate("artifact_families", fam_doc)
    fams = Families(fam_doc)
    src_fd = io.open(str(src), DIR_FLAGS)
    try:
        tree = _source_tree(io, src_fd, counter)
        shards = list_shards(io, src_fd)
        if len(shards) > lim["max_files"]:
            raise ArtifactError("too_many_files", f"{len(shards)} shards > {lim['max_files']}")
        files, classified, total = [], [], 0
        for name in shards:
            fd, st = open_regular(io, name, src_fd)
            try:
                prefix = read_header_prefix(io, fd, st.st_size, lim)
            finally:
                io.close(fd)
            counter["metadata_bytes_read"] += len(prefix)
            h = parse_header(prefix, st.st_size, lim)
            total += len(h["tensors"])
            if total > lim["max_total_tensors"]:
                raise ArtifactError("too_many_tensors", f"more than {lim['max_total_tensors']} tensors")
            for t in h["tensors"]:
                classified.append((fams.classify(t), t))
            check_expert_metadata(h["metadata"], h["tensors"])
            entry = tree["files"].get(name) if tree else None
            if entry is None:
                upstream = None
            elif "lfs_sha256" in entry:
                upstream = {"lfs_sha256": entry["lfs_sha256"], "size": entry["size"]}
            else:
                upstream = {"blob_id": entry["blob_id"], "size": entry["size"]}
            files.append({"path": name, "bytes": st.st_size, "header_bytes": h["header_bytes"],
                          "header_sha256": h["header_sha256"], "tensors": len(h["tensors"]), "upstream": upstream})
        fams.check_groups(classified)
    finally:
        io.close(src_fd)
    families = fams.tally(classified)
    sums = family_totals(families)
    up = [f["upstream"] for f in files]
    inv = {
        "schema_version": "1",
        "family_rules": {"rules_id": fam_doc["rules_id"], "sha256": hashlib.sha256(fam_raw).hexdigest()},
        "source": None if tree is None else {"revision": tree["revision"], "tree_sha256": tree["tree_sha256"]},
        "header_digest": header_digest([(f["path"], f["header_sha256"]) for f in files]),
        "files": files,
        "families": families,
        "by_dtype": sums["by_dtype"],
        "by_placement": sums["by_placement"],
        "totals": {"files": len(files), "tensors": len(classified), "file_bytes": sum(f["bytes"] for f in files),
                   "header_bytes": sum(f["header_bytes"] for f in files),
                   "data_bytes": sum(f["bytes"] - f["header_bytes"] for f in files),
                   "misaligned": sum(f["misaligned"] for f in families)},
        "upstream": {"lfs_files": sum(u is not None and "lfs_sha256" in u for u in up),
                     "lfs_size_match": sum(u is not None and "lfs_sha256" in u and u["size"] == f["bytes"]
                                           for u, f in zip(up, files)),
                     "non_lfs_files": sum(u is not None and "blob_id" in u for u in up),
                     "not_in_tree": sum(u is None for u in up)},
    }
    validate("artifact_inventory", inv)
    return inv


def dump_header(path: Path, *, limits: dict | None = None, io: FileIO | None = None) -> dict:
    io = io or FileIO()
    lim = limits_with(limits)
    path = Path(path)
    dfd = io.open(str(path.parent), DIR_FLAGS & ~os.O_NOFOLLOW)
    try:
        fd, st = open_regular(io, path.name, dfd)
    finally:
        io.close(dfd)
    try:
        prefix = read_header_prefix(io, fd, st.st_size, lim)
    finally:
        io.close(fd)
    return parse_header(prefix, st.st_size, lim)


def reject(exc: Exception) -> int:
    cls = exc.cls if isinstance(exc, ArtifactError) else "error"
    msg = str(exc)
    if isinstance(exc, ArtifactError) and msg.startswith(cls + ": "):
        msg = msg[len(cls) + 2:]
    sys.stderr.write(f"{cls}: {msg}\n")
    return 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, nargs="?")
    p.add_argument("--families", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--dump-header", type=Path, metavar="SHARD")
    p.add_argument("--limits", type=json.loads, default=None, help="JSON object overriding Q15 limits")
    args = p.parse_args(argv)
    if (args.dump_header is None) == (args.src is None) or (args.src is not None and args.families is None):
        p.error("give SRC --families F, or --dump-header SHARD")
    try:
        value = (dump_header(args.dump_header, limits=args.limits) if args.dump_header is not None
                 else inventory(args.src, args.families, limits=args.limits))
        data = canonical_json(value)
    except (ValueError, OSError) as e:
        return reject(e)
    if args.out:
        args.out.write_bytes(data)
    else:
        sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
