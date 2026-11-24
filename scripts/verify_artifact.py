"""Read-only reference inspector for prepared v2 artifact directories (M2.1 Q4, Q7, Q15, Q16, Q17).

Every file is opened once with `openat(dirfd, name, O_RDONLY|O_NOFOLLOW)` and held; checks run in the contract
order shared with cerec (tests/fixtures/artifact_invalid.json). Modes:
  headers  structure, shard headers and families; reads no payload byte;
  stamp    headers + the sidecar stamp `<parent>/<dir>.verified.json` (stamp_source full_verify) compared with
           fstat/fstatvfs of the held descriptors; reads no payload byte;
  full     headers + sha256 of every listed file.
`python3 scripts/verify_artifact.py DIR --expected-manifest-sha256 H --mode full|stamp|headers` prints a JSON report;
rejections exit 2 with `<error_class>: <message>` as the first stderr line. Never writes anything.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from cere_contracts import (ArtifactError, ContractError, canonical_json, family_totals, limits_with, parse_json,
                            validate)
from model_inventory import (DIR_FLAGS, Families, FileIO, check_expert_metadata, open_regular, parse_header,
                             read_header_prefix, reject)

MODES = ("full", "stamp", "headers")
HASH_CHUNK = 8 << 20
# Stamp format 2 (owner 2026-10-15) names a file's filesystem by the UUID-derived `fsid` (16 hex digits of f_fsid)
# or, without one, by `dev`; a stamp without a `format` key is format 1 and names every file by `dev`.
STAMP_FORMAT = 2
STAMP_FIELDS = ("ino", "size", "mtime_ns", "ctime_ns", "mode")


def stamp_path(directory: Path) -> Path:
    return directory.parent / f"{directory.name}.verified.json"


def stamp_facts(st: os.stat_result) -> dict:
    return {"ino": st.st_ino, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
            "mode": st.st_mode}


def fsid_hex(fd: int) -> str:
    return f"{os.fstatvfs(fd).f_fsid:016x}"


class _Artifact:
    def __init__(self, io: FileIO, limits: dict):
        self.io = io
        self.limits = limits
        self.fds: list[int] = []
        self.counter = {"payload_bytes_read": 0, "metadata_bytes_read": 0}

    def opendir(self, name: str, dir_fd: int | None) -> int:
        try:
            fd = self.io.open(name, DIR_FLAGS, dir_fd=dir_fd)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise ArtifactError("symlink", f"{name} is a symlink") from None
            if e.errno == errno.ENOTDIR:
                raise ArtifactError("special_file", f"{name} is not a directory") from None
            raise
        self.fds.append(fd)
        return fd

    def open_file(self, path: str, root_fd: int):
        """Open a listed (possibly nested) path without following any symlink; the descriptor is held."""
        parts = path.split("/")
        fd = root_fd
        for part in parts[:-1]:
            fd = self.opendir(part, fd)
        f, st = open_regular(self.io, parts[-1], fd)
        self.fds.append(f)
        return f, st

    def close(self):
        for fd in reversed(self.fds):
            self.io.close(fd)
        self.fds.clear()


def _walk(art: _Artifact, dir_fd: int, prefix: str, listed: set, dirs: set) -> None:
    """Step 7: everything under the root is a listed file, the manifest pair, or an ancestor dir of a listed file."""
    for name in sorted(art.io.listdir(dir_fd), key=str.encode):
        rel = prefix + name
        st = art.io.lstat(name, dir_fd)
        if stat.S_ISLNK(st.st_mode):
            raise ArtifactError("symlink", f"{rel} is a symlink")
        if stat.S_ISDIR(st.st_mode):
            if rel not in dirs:
                raise ArtifactError("unlisted_file", f"{rel}/ is not listed")
            _walk(art, art.opendir(name, dir_fd), rel + "/", listed, dirs)
        elif stat.S_ISREG(st.st_mode):
            if rel not in listed and rel not in ("manifest.json", "manifest.sha256"):
                raise ArtifactError("unlisted_file", f"{rel} is not listed")
        else:
            raise ArtifactError("special_file", f"{rel} is not a regular file or directory")


def _read_manifest(art: _Artifact, root_fd: int, expected_sha256: str) -> tuple[dict, bytes]:
    fd, st = art.open_file("manifest.json", root_fd)
    if st.st_size > art.limits["max_manifest_bytes"]:
        raise ArtifactError("manifest_size", f"manifest is {st.st_size} bytes > {art.limits['max_manifest_bytes']}")
    raw = art.io.pread(fd, st.st_size, 0)
    art.counter["metadata_bytes_read"] += len(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ArtifactError("manifest_digest", f"manifest sha256 {actual} != expected {expected_sha256}")
    try:
        manifest = parse_json(raw)  # parse exactly the bytes that were verified
    except ArtifactError:
        raise
    except ValueError as e:
        raise ArtifactError("manifest_schema", f"manifest is not JSON: {e}") from None
    try:
        canonical = canonical_json(manifest)
    except ContractError as e:
        raise ArtifactError("manifest_noncanonical", str(e)) from None
    if canonical != raw:
        raise ArtifactError("manifest_noncanonical", "manifest bytes are not the Q17 canonical encoding")
    try:
        validate("execution_artifact", manifest, limits=art.limits)
    except ArtifactError:
        raise
    except ContractError as e:
        raise ArtifactError("manifest_schema", str(e)) from None
    return manifest, raw


def _check_stamp(art: _Artifact, directory: Path, manifest_sha256: str, held: dict) -> None:
    path = stamp_path(directory)
    try:
        dfd = art.io.open(str(path.parent), DIR_FLAGS & ~os.O_NOFOLLOW)
    except OSError as e:
        raise ArtifactError("stamp", f"cannot open stamp directory: {e}") from None
    art.fds.append(dfd)
    try:
        fd, st = art.open_file(path.name, dfd)
        raw = art.io.pread(fd, st.st_size, 0)
        art.counter["metadata_bytes_read"] += len(raw)
        stamp = parse_json(raw)
    except (OSError, ValueError) as e:
        raise ArtifactError("stamp", f"no readable stamp {path}: {e}") from None
    if not isinstance(stamp, dict) or stamp.get("stamp_source") != "full_verify":
        raise ArtifactError("stamp", "full verify required: stamp_source is not full_verify")
    if stamp.get("manifest_sha256") != manifest_sha256:
        raise ArtifactError("stamp", "stamp is for another manifest")
    files = stamp.get("files")
    if not isinstance(files, dict) or set(files) != set(held):
        raise ArtifactError("stamp", "stamp file set differs from the manifest")
    legacy = "format" not in stamp
    if not legacy and stamp["format"] != STAMP_FORMAT:
        raise ArtifactError("stamp", f"full verify required: unknown stamp format {stamp['format']!r}")
    # A record is exactly the fstat fields plus `fsid` (format 2 only) or `dev`, as cerec's stamp parser takes it.
    shapes = [{*STAMP_FIELDS, "dev"}] + ([] if legacy else [{*STAMP_FIELDS, "fsid"}])
    for p, (fd, st) in held.items():
        rec = files[p]
        if not isinstance(rec, dict) or set(rec) not in shapes:
            raise ArtifactError("stamp", f"full verify required: {p} record is malformed")
        if {k: rec[k] for k in STAMP_FIELDS} != stamp_facts(st):
            raise ArtifactError("stamp", f"full verify required: {p} changed since the stamp")
        if "fsid" in rec:
            same_fs = rec["fsid"] == fsid_hex(fd)
        else:
            same_fs = rec["dev"] == st.st_dev
            if not same_fs and legacy:
                raise ArtifactError("stamp", f"{p}: st_dev differs and the stamp predates fs-uuid format; "
                                    "full verify required")
        if not same_fs:
            raise ArtifactError("stamp", f"full verify required: {p} changed since the stamp")


def inspect_artifact(directory: Path, expected_manifest_sha256: str, *, mode: str = "full",
                     limits: dict | None = None, io: FileIO | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    directory = Path(directory)
    art = _Artifact(io or FileIO(), limits_with(limits))
    try:
        if directory.is_symlink():
            raise ArtifactError("symlink", "artifact root is a symlink")
        root_fd = art.opendir(str(directory), None)
        manifest, raw = _read_manifest(art, root_fd, expected_manifest_sha256)
        files = manifest["files"]
        listed = {f["path"] for f in files}
        dirs = {"/".join(p.split("/")[:i]) for p in listed for i in range(1, p.count("/") + 1)}
        _walk(art, root_fd, "", listed, dirs)
        held = {}
        for f in files:
            fd, st = art.open_file(f["path"], root_fd)
            if st.st_size != f["bytes"]:
                raise ArtifactError("file_size", f"{f['path']} is {st.st_size} bytes, manifest says {f['bytes']}")
            held[f["path"]] = (fd, st)
        shards, total = [], 0
        for f in files:
            if f["role"] != "weights":
                continue
            fd, st = held[f["path"]]
            prefix = read_header_prefix(art.io, fd, st.st_size, art.limits)
            art.counter["metadata_bytes_read"] += len(prefix)
            sha = hashlib.sha256(prefix).hexdigest()
            if (len(prefix), sha) != (f["safetensors"]["header_bytes"], f["safetensors"]["header_sha256"]):
                raise ArtifactError("header_hash", f"{f['path']}: header differs from the manifest")
            h = parse_header(prefix, st.st_size, art.limits)
            total += len(h["tensors"])
            if total > art.limits["max_total_tensors"]:
                raise ArtifactError("too_many_tensors", f"more than {art.limits['max_total_tensors']} tensors")
            shards.append((f, prefix, h))
        fams = Families(manifest["family_rules"], manifest["profile"]["placement"])
        classified = []
        for _f, _prefix, h in shards:
            for t in h["tensors"]:
                classified.append((fams.classify(t), t))
        for _f, _prefix, h in shards:
            check_expert_metadata(h["metadata"], h["tensors"])
        fams.check_groups(classified)
        for f, _prefix, h in shards:
            if len(h["tensors"]) != f["safetensors"]["tensors"]:
                raise ArtifactError("family_totals", f"{f['path']}: tensor count differs from the manifest")
        families = fams.tally(classified)
        if families != manifest["families"]:
            bad = next(a["family"] for a, b in zip(families, manifest["families"]) if a != b)
            raise ArtifactError("family_totals", f"family {bad}: headers differ from the manifest counters")
        if mode == "stamp":
            _check_stamp(art, directory, hashlib.sha256(raw).hexdigest(), held)
        if mode == "full":
            prefixes = {f["path"]: prefix for f, prefix, _h in shards}
            for f in files:
                fd, st = held[f["path"]]
                h = hashlib.sha256()
                pos = 0
                if f["path"] in prefixes:  # the header region was already read once (metadata)
                    h.update(prefixes[f["path"]])
                    pos = len(prefixes[f["path"]])
                while pos < st.st_size:
                    chunk = art.io.pread(fd, min(HASH_CHUNK, st.st_size - pos), pos)
                    if not chunk:
                        raise ArtifactError("file_size", f"{f['path']} shrank while hashing")
                    h.update(chunk)
                    pos += len(chunk)
                    art.counter["payload_bytes_read"] += len(chunk)
                if h.hexdigest() != f["sha256"]:
                    raise ArtifactError("file_hash", f"{f['path']}: sha256 differs from the manifest")
        totals = family_totals(families)
        return {"outcome": "verified", "mode": mode, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "profile_id": manifest["profile_id"], "files": len(files), "shards": len(shards),
                "tensors": len(classified), "families": families, "by_dtype": totals["by_dtype"],
                "by_placement": totals["by_placement"],
                "misaligned": sum(f["misaligned"] for f in families), **art.counter}
    finally:
        art.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("directory", type=Path)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--limits", type=json.loads, default=None, help="JSON object overriding Q15 limits")
    args = p.parse_args(argv)
    try:
        result = inspect_artifact(args.directory, args.expected_manifest_sha256, mode=args.mode, limits=args.limits)
    except (ValueError, OSError) as e:
        return reject(e)
    sys.stdout.buffer.write(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
