#!/usr/bin/env python3
"""SelfTune ST4 S0/S1 (tasks/selftune-spec.md ST4, "ST4 S0 source pin"): the pinned source file set of a second model.

usage: python3 scripts/native/model_source.py SOURCE_JSON DIR [--mode download|execute|prepare] [--hash-weights]
    [--write-tree] [--derived RECORD]

SOURCE_JSON (e.g. tests/fixtures/model_configs/dice-8b/source.json) pins every file fetched from the hub at one
revision (`files`: name -> {sha256, bytes, role}) and the authors' decode loop (`decode_loop`: one file of another
repo at a commit). Roles:
- `code`: remote code reviewed by the maintainer; the only hub `.py` files a reference run may import;
- `never_execute`: remote code fetched for the record only (the tokenizer files: the reference uses transformers'
  built-in class);
- `config`, `tokenizer`, `doc`: data;
- `weights`: safetensors shards (sha256 = the hub's LFS oid).
A file name is a plain name, or `<dir>/<plain name>` for a directory of SUBDIRS (Inkling 2 Mini,
tasks/inkling-spec.md S0: `tiktoken/tokenizer.model`); any other directory is refused, in the pin and in a checked
directory alike. An entry carries `lfs: true` when the hub stores a non-weights file in LFS (Inkling 2 Mini's
`tokenizer.json` and `tiktoken/tokenizer.model`; its sha256 is then the LFS oid, as a weights file's always is).

`verify` refuses a directory that differs from the pin, by name:
- `download` mode (the hub mirror, e.g. /var/lib/cere/models/DICE-8B@ca80967e): every pinned file is present, with its
  size and sha256, and nothing else is;
- `execute` mode (the directory a reference container loads with trust_remote_code): the same, except that the
  `never_execute` files must be absent and the decode loop file must be present. The only `.py` files allowed are the
  `code` set plus the decode loop, so trust_remote_code cannot reach any other code;
- `prepare` mode (the directory `cerec prepare` reads, ST4 S3b, e.g. /var/lib/cere/models/DICE-8B@ca80967e-prepare, a
  hardlinked copy of the mirror): the download set plus the HF tree metadata `cerec prepare` requires,
  `.cache/huggingface/trees/<revision>.json`, equal byte for byte to the tree rendered from the pin (`tree_bytes`:
  every pinned file with its size and git blob id; a weights file is LFS, so its blob id is the one of its LFS
  pointer and it carries `lfs_sha256` = its pinned sha256). `--write-tree` writes that file first when it is absent.
  `--derived RECORD` (ST4 S3d, D-m) also admits the files a derived-files record names (DICE-8B's `tokenizer.json`,
  converted by the reference image), each with its recorded size and sha256; they are not in the tree file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROLES = ("code", "never_execute", "config", "tokenizer", "doc", "weights")
SUBDIRS = ("tiktoken",)
MODES = ("download", "execute", "prepare")
TREE_DIR = (".cache", "huggingface", "trees")
SHA256_HEX = 64


class SourceError(ValueError):
    """The source pin or a directory checked against it is refused."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _entry(where: str, entry: object) -> None:
    keys = set(entry) if isinstance(entry, dict) else set()
    if not isinstance(entry, dict) or not {"sha256", "bytes", "role"} <= keys <= {"sha256", "bytes", "role", "lfs"}:
        raise SourceError(f"{where}: an entry is exactly {{sha256, bytes, role}} (and lfs)")
    sha, size = entry.get("sha256"), entry.get("bytes")
    if not (isinstance(sha, str) and len(sha) == SHA256_HEX and all(c in "0123456789abcdef" for c in sha)):
        raise SourceError(f"{where}: sha256 is not 64 lowercase hex digits")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SourceError(f"{where}: bytes is not a positive integer")
    if "lfs" in entry and (entry["lfs"] is not True or entry["role"] == "weights"):
        raise SourceError(f"{where}: lfs is only `true`, on a non-weights file (weights are always LFS)")


def _pinned_name(name: str) -> bool:
    """A plain file name, or `<dir>/<plain name>` under a directory of SUBDIRS."""
    parts = name.split("/")
    plain = parts[-1] not in ("", ".", "..")
    return plain and (len(parts) == 1 or (len(parts) == 2 and parts[0] in SUBDIRS))


def load_source(path: Path) -> dict:
    """The pin, after its own rules: well-formed entries, known roles, plain file names, the config sha256 equal to
    the top-level `sha256` (what the descriptor check reads), and a decode loop that is a `.py` file."""
    source = json.loads(Path(path).read_text())
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        raise SourceError(f"{path}: files is absent or empty")
    for name, entry in files.items():
        if not _pinned_name(name):
            raise SourceError(f"{path}: files {name!r} is not a plain file name (or one under {', '.join(SUBDIRS)}/)")
        _entry(f"{path}: files {name}", entry)
        if entry["role"] not in ROLES:
            raise SourceError(f"{path}: files {name} role {entry['role']!r} is not one of {', '.join(ROLES)}")
        if name.endswith(".py") != (entry["role"] in ("code", "never_execute")):
            raise SourceError(f"{path}: files {name} role {entry['role']} does not match its kind (.py is code)")
    config = files.get("config.json")
    if config is None or config["sha256"] != source.get("sha256"):
        raise SourceError(f"{path}: files config.json sha256 differs from the top-level sha256")
    loop = source.get("decode_loop")
    if not isinstance(loop, dict):
        raise SourceError(f"{path}: decode_loop is absent")
    _entry(f"{path}: decode_loop", {k: loop.get(k) for k in ("sha256", "bytes")} | {"role": "code"})
    name = loop.get("file")
    if not isinstance(name, str) or "/" in name or not name.endswith(".py") or name in files:
        raise SourceError(f"{path}: decode_loop file {name!r} is not a plain .py name apart from files")
    return source


def expected(source: dict, mode: str) -> dict:
    """name -> entry of the files a directory in `mode` holds."""
    if mode not in MODES:
        raise SourceError(f"mode {mode!r} is not one of {', '.join(MODES)}")
    want = {n: e for n, e in source["files"].items() if mode != "execute" or e["role"] != "never_execute"}
    if mode == "execute":
        loop = source["decode_loop"]
        want[loop["file"]] = {"sha256": loop["sha256"], "bytes": loop["bytes"], "role": "code"}
    return want


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lfs_pointer(entry: dict) -> bytes:
    """The git-lfs pointer file the hub's git tree holds for an LFS file of `entry` (sha256, bytes)."""
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{entry['sha256']}\nsize {entry['bytes']}\n".encode()


def tree_bytes(source: dict, directory: Path) -> bytes:
    """The HF tree metadata of the pinned files (`cerec prepare`'s `.cache/huggingface/trees/<revision>.json`,
    format of `source::parse_tree`): name -> {size, blob_id, lfs_sha256 for weights}. A non-LFS blob id is read
    from `directory`, whose bytes `verify` has tied to the pin's sha256."""
    files = {}
    for name, entry in sorted(source["files"].items()):
        if entry["role"] == "weights" or entry.get("lfs"):
            files[name] = {"blob_id": _git_blob_id(lfs_pointer(entry)), "lfs_sha256": entry["sha256"],
                           "size": entry["bytes"]}
        else:
            files[name] = {"blob_id": _git_blob_id((Path(directory) / name).read_bytes()), "size": entry["bytes"]}
    return (json.dumps({"files": files}, indent=1, sort_keys=True) + "\n").encode()


def tree_path(source: dict, directory: Path) -> Path:
    rev = source.get("revision")
    if not (isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)):
        raise SourceError("the pin's revision is not 40 lowercase hex digits (the tree file is named after it)")
    return Path(directory).joinpath(*TREE_DIR, f"{rev}.json")


def _check_tree(source: dict, directory: Path) -> None:
    """`prepare` mode: `.cache` holds exactly huggingface/trees/<revision>.json, equal to `tree_bytes`."""
    node = Path(directory)
    for i, part in enumerate(TREE_DIR):
        node = node / part
        if node.is_symlink() or not node.is_dir():
            raise SourceError(f"{directory}: {'/'.join(TREE_DIR[:i + 1])} is not a directory (prepare mode)")
        inside = sorted(p.name for p in node.iterdir())
        want = [TREE_DIR[i + 1]] if i + 1 < len(TREE_DIR) else [tree_path(source, directory).name]
        if inside != want:
            raise SourceError(f"{directory}: {'/'.join(TREE_DIR[:i + 1])} holds {inside}, not {want} (prepare mode)")
    p = tree_path(source, directory)
    if p.is_symlink() or not p.is_file() or p.read_bytes() != tree_bytes(source, directory):
        raise SourceError(f"{directory}: {'/'.join(TREE_DIR)}/{p.name} differs from the tree rendered from the pin")


def load_derived(path: Path) -> dict:
    """name -> entry of a derived-files record (ST4 S3d, D-m: e.g. tests/fixtures/model_configs/dice-8b/derived.json,
    the `tokenizer.json` the hub source lacks): data files only (no code, no weights), plain names."""
    files = json.loads(Path(path).read_text()).get("files")
    if not isinstance(files, dict) or not files:
        raise SourceError(f"{path}: files is absent or empty")
    for name, entry in files.items():
        if "/" in name or name in ("", ".", "..") or name.endswith(".py"):
            raise SourceError(f"{path}: files {name!r} is not a plain data file name")
        _entry(f"{path}: files {name}", entry)
        if entry["role"] not in ("config", "tokenizer", "doc"):
            raise SourceError(f"{path}: files {name} role {entry['role']!r} is not config, tokenizer or doc")
    return files


def verify(source: dict, directory: Path, mode: str, *, hash_weights: bool = False, write_tree: bool = False,
           derived: dict | None = None) -> int:
    """Refuse `directory` unless it holds exactly the pinned files of `mode` (see the module doc), plus in `prepare`
    mode the `derived` files (`load_derived`), each with its recorded size and sha256 (they are not in the tree
    file). With `write_tree` (`prepare` mode only) an absent tree file is written once the pinned files pass.
    Returns the number of files checked."""
    directory = Path(directory)
    want = expected(source, mode)
    if write_tree and mode != "prepare":
        raise SourceError(f"--write-tree needs prepare mode, not {mode}")
    if derived:
        if mode != "prepare":
            raise SourceError(f"derived files need prepare mode, not {mode}")
        clash = sorted(set(derived) & set(want))
        if clash:
            raise SourceError(f"derived files {clash} are pinned source files")
        want.update(derived)
    if not directory.is_dir():
        raise SourceError(f"{directory}: not a directory")
    present = {}
    for p in directory.iterdir():
        if mode == "prepare" and p.name == TREE_DIR[0]:
            continue
        if p.name in SUBDIRS and p.is_dir() and not p.is_symlink():
            if not any(n.startswith(p.name + "/") for n in want):
                raise SourceError(f"{directory}: {p.name}/ is an unpinned directory ({mode} mode)")
            present.update({f"{p.name}/{q.name}": q for q in p.iterdir()})
        else:
            present[p.name] = p
    for name, p in sorted(present.items()):
        if name not in want:
            what = "remote code" if name.endswith(".py") else "file"
            raise SourceError(f"{directory}: {name} is an unpinned {what} ({mode} mode)")
        if not p.is_file() or p.is_symlink():
            raise SourceError(f"{directory}: {name} is not a regular file")
    for name, entry in sorted(want.items()):
        p = present.get(name)
        if p is None:
            raise SourceError(f"{directory}: {name} is missing ({mode} mode)")
        size = p.stat().st_size
        if size != entry["bytes"]:
            raise SourceError(f"{directory}: {name} has {size} bytes, the pin {entry['bytes']}")
        if entry["role"] != "weights" or hash_weights:
            got = _sha256(p)
            if got != entry["sha256"]:
                raise SourceError(f"{directory}: {name} sha256 {got} differs from the "
                                  f"{'derived record' if derived and name in derived else 'pin'} {entry['sha256']}")
    if mode == "prepare":
        tree = tree_path(source, directory)
        if write_tree and not tree.exists() and not directory.joinpath(TREE_DIR[0]).exists():
            tree.parent.mkdir(parents=True)
            tree.write_bytes(tree_bytes(source, directory))
        _check_tree(source, directory)
    return len(want)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--mode", choices=MODES, default="download")
    ap.add_argument("--hash-weights", action="store_true")
    ap.add_argument("--write-tree", action="store_true", help="prepare mode: write an absent tree file first")
    ap.add_argument("--derived", type=Path, help="prepare mode: a derived-files record (D-m)")
    args = ap.parse_args(argv)
    try:
        derived = load_derived(args.derived) if args.derived else None
        n = verify(load_source(args.source), args.directory, args.mode, hash_weights=args.hash_weights,
                   write_tree=args.write_tree, derived=derived)
    except (SourceError, OSError, json.JSONDecodeError) as e:
        print(f"model_source: refused: {e}", file=sys.stderr)
        return 1
    print(f"model_source: {args.directory}: {n} files equal the pin ({args.mode} mode"
          f"{', weights hashed' if args.hash_weights else ', weights by size'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
