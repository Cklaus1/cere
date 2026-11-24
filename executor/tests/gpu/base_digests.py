"""O186 (DevLoop3): the cache of K8Forward's merge-base digests (torch-free: tests/test_base_digests.py imports it,
native_dev.py loads it by path).

K8Forward's L1/L2 rows (test_model_k8_forward.py) compare this build's forward_trace and decode-invariance digests with
the merge-base library's (`merge_base_digests`: {library_sha256, desc_size, trace {prompt id: sha256}, decode {case:
sha256}}), which cost a model load of their own under BASE_KTEST_LIB and its traces. They are a function of their
inputs only, so they are kept in a directory, one JSON entry per key: the SHA-256 of the inputs (KEY_FIELDS)
    library_sha256     the merge-base libcere_ktest.so's bytes
    manifest_sha256    the pin's manifest_sha256, once the pinned artifact's manifest.json (mounted at MANIFEST_MOUNT)
                       hashes to it and names /model's weights files (`pinned`)
    prompts_sha256     the prefill prompts as run: each (prompt id, token ids), in order (`prompts_sha256`)
    harness_sha256     the bytes of the test module, every local module it imports, transitively, and the extra files
                       the digests read (the families file) (`harness_sha256`)
    runtime            the torch and CUDA versions and the device name (L2's tokens come from torch's generator)
`native_dev.py [--gpu 1] test --suite S [--base-digests MODE]` (S discovering test_model_k8_forward.py: model_k8,
model) mounts the host cache directory read-write and the pinned artifact's manifest.json read-only at MANIFEST_MOUNT
(/model, the checkpoint, holds no manifest of its own), and sets:
    CERE_BASE_DIGESTS=cached|fresh|verify   MODE (default cached)
    CERE_BASE_DIGESTS_DIR=<dir>             the cache directory in the container
    CERE_BASE_DIGESTS_MANIFEST=<64 hex>     manifest_sha256
Modes (`resolve`): `cached` uses a valid entry (source `cached`) or computes fresh and stores it (source `fresh`);
`fresh` computes and stores (replacing any entry); `verify` computes fresh and raises BaseDigestsError unless a valid
entry exists and equals it, field by field (source `fresh`, verified). An unreadable, malformed or mismatched entry
(another schema, key or input set; digests of another library or prompt set) is a miss. A result is stored only when
the computation returned and has the digests' shape (`check_digests`), through a temporary file in the directory renamed
over the entry (atomic: a reader sees the old entry or the new one). A failed store leaves the run's fresh digests in
use (stored false). Without CERE_BASE_DIGESTS_DIR and CERE_BASE_DIGESTS_MANIFEST (a run outside native_dev.py) nothing
is read or stored: `cached` and `fresh` compute fresh, `verify` raises.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

MODE_ENV = "CERE_BASE_DIGESTS"
DIR_ENV = "CERE_BASE_DIGESTS_DIR"
MANIFEST_ENV = "CERE_BASE_DIGESTS_MANIFEST"
MANIFEST_MOUNT = Path("/digest-cache-manifest.json")  # the pinned artifact's manifest.json in the container
MODES = ("cached", "fresh", "verify")
DEFAULT_MODE = "cached"
SCHEMA = "cere-base-digests-v1"
KEY_FIELDS = ("library_sha256", "manifest_sha256", "prompts_sha256", "harness_sha256", "runtime")
DIGEST_FIELDS = ("library_sha256", "desc_size", "trace", "decode")
HEX64 = re.compile(r"[0-9a-f]{64}")
TMP_PREFIX = ".tmp-"
ENTRY_MODE = 0o644


class BaseDigestsError(AssertionError):
    """A cache configuration, verify or digest-shape failure: the calling test fails (as oracle.OracleError)."""


class Config(NamedTuple):
    mode: str
    directory: Path | None  # None: no cache (nothing read or stored)
    manifest_sha256: str | None


class Result(NamedTuple):
    digests: dict
    source: str  # "cached" (a valid entry) or "fresh" (computed in this run)
    key: str | None  # None without a cache
    stored: bool  # this run wrote the entry
    verified: bool  # verify mode: the fresh digests equal the cached entry's


def config(env: Mapping[str, str]) -> Config:
    """The cache configuration of `env` (module docstring); BaseDigestsError on an unknown mode, a relative directory,
    a malformed manifest digest or only one of DIR_ENV and MANIFEST_ENV."""
    mode = env.get(MODE_ENV, DEFAULT_MODE)
    if mode not in MODES:
        raise BaseDigestsError(f"{MODE_ENV}={mode!r} is not one of {'|'.join(MODES)}")
    directory, manifest = env.get(DIR_ENV), env.get(MANIFEST_ENV)
    if (directory is None) != (manifest is None):
        raise BaseDigestsError(f"{DIR_ENV} and {MANIFEST_ENV} go together (native_dev.py sets both)")
    if directory is None:
        return Config(mode, None, None)
    if not Path(directory).is_absolute():
        raise BaseDigestsError(f"{DIR_ENV}={directory!r} is not an absolute path")
    if not HEX64.fullmatch(manifest):
        raise BaseDigestsError(f"{MANIFEST_ENV}={manifest!r} is not a sha256")
    return Config(mode, Path(directory), manifest)


def pinned(cfg: Config, manifest: Path, model_dir: Path) -> Config:
    """`cfg`, or `cfg` without its cache unless `manifest` (the pinned artifact's manifest.json as mounted, at
    MANIFEST_MOUNT) hashes to cfg.manifest_sha256 (the pin's, from native_dev.py) and names `model_dir` (the checkpoint
    the merge-base digests read; `model_files`). Otherwise a miss, computed fresh with nothing read or stored (`verify`
    then raises in `resolve`)."""
    if cfg.directory is None:
        return cfg
    try:
        raw = manifest.read_bytes()
    except OSError as e:
        reason = f"cannot read {manifest} ({e!r})"
    else:
        if hashlib.sha256(raw).hexdigest() != cfg.manifest_sha256:
            reason = f"{manifest} does not hash to the pin's manifest_sha256"
        elif (reason := model_files(raw, model_dir)) is None:
            return cfg
    print(f"base_digests: no cache: {reason}; the merge-base digests are computed fresh", file=sys.stderr, flush=True)
    return cfg._replace(directory=None, manifest_sha256=None)


def model_files(raw: bytes, model_dir: Path) -> str | None:
    """None when the manifest `raw` names `model_dir`'s weights: it lists at least one weights file, each is the
    checkpoint's (upstream.lfs_sha256 == sha256, the dev-loop share's condition) and a file of its byte length under
    `model_dir` (a stat each, no hashing); else why not."""
    try:
        weights = [f for f in json.loads(raw)["files"] if f["role"] == "weights"]
        for f in weights:
            name, size = f["path"], f["bytes"]
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                return f"the manifest names weights file {name!r} outside the checkpoint"
            if f.get("upstream", {}).get("lfs_sha256") != f["sha256"]:
                return f"the manifest's {name} is not the checkpoint's (upstream.lfs_sha256 != sha256)"
            path = model_dir / name
            if not path.is_file() or path.stat().st_size != size:
                return f"{path} is not the manifest's {name} ({size} bytes)"
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return f"the manifest is malformed ({e!r})"
    except OSError as e:
        return f"cannot stat the checkpoint's weights ({e!r})"
    if not weights:
        return "the manifest lists no weights file"
    return None


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prompts_sha256(prompts: Sequence[tuple[str, Sequence[int]]]) -> str:
    """SHA-256 of the prompts as run: each (prompt id, token ids), in order."""
    return _sha256_json([[pid, [int(t) for t in ids]] for pid, ids in prompts])


def local_modules(module: Path) -> list[Path]:
    """`module` and every module beside it that it imports, transitively (`import x`, `from x import ...` and
    `importlib.import_module("x")`, at any depth of the source), sorted by name."""
    seen: dict[str, Path] = {}
    todo = [module.resolve()]
    while todo:
        path = todo.pop()
        if path.name in seen:
            continue
        seen[path.name] = path
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.level == 0 else []
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "import_module" and node.args and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            else:
                continue
            todo += [path.parent / f"{n}.py" for n in names if (path.parent / f"{n}.py").is_file()]
    return [seen[name] for name in sorted(seen)]


def harness_sha256(module: Path, extra: Sequence[Path] = ()) -> str:
    """SHA-256 over (file name, byte length, bytes) of `local_modules(module)`, then of each `extra` file in order."""
    h = hashlib.sha256()
    for path in [*local_modules(module), *extra]:
        data = Path(path).read_bytes()
        h.update(f"{Path(path).name}\0{len(data)}\0".encode() + data)
    return h.hexdigest()


def entry_key(inputs: Mapping[str, str]) -> str:
    """The cache key: SHA-256 of SCHEMA and `inputs` (exactly KEY_FIELDS, each a non-empty string)."""
    if sorted(inputs) != sorted(KEY_FIELDS) or not all(isinstance(v, str) and v for v in inputs.values()):
        raise BaseDigestsError(f"cache inputs {sorted(inputs)} are not the non-empty strings {list(KEY_FIELDS)}")
    return _sha256_json({"schema": SCHEMA, **inputs})


def check_digests(digests: object, library_sha256: str, prompt_ids: Sequence[str]) -> None:
    """BaseDigestsError unless `digests` is merge_base_digests' shape for this library and prompt set: exactly
    DIGEST_FIELDS; library_sha256 the given one; desc_size a positive int; trace {each prompt id: sha256}; decode a
    non-empty {case: sha256}."""
    def hashes(value) -> bool:
        return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) and HEX64.fullmatch(v)
                                               for k, v in value.items())
    if not isinstance(digests, dict) or sorted(digests) != sorted(DIGEST_FIELDS):
        raise BaseDigestsError(f"merge-base digests are not {{{', '.join(DIGEST_FIELDS)}}}")
    if digests["library_sha256"] != library_sha256:
        raise BaseDigestsError(f"merge-base digests are of library {digests['library_sha256']!r}, not {library_sha256}")
    size = digests["desc_size"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise BaseDigestsError(f"merge-base desc_size {size!r} is not a positive int")
    if not hashes(digests["trace"]) or sorted(digests["trace"]) != sorted(prompt_ids):
        raise BaseDigestsError(f"merge-base trace digests are not one sha256 per prompt of {list(prompt_ids)}")
    if not hashes(digests["decode"]) or not digests["decode"]:
        raise BaseDigestsError("merge-base decode digests are not a non-empty {case: sha256}")


def entry_path(directory: Path, key: str) -> Path:
    return directory / f"{key}.json"


def load(directory: Path, key: str, inputs: Mapping[str, str], prompt_ids: Sequence[str]) -> dict | None:
    """The digests of `key`'s entry, or None (a miss) when it is absent, unreadable, or not this schema, key and
    input set with digests of `check_digests`' shape."""
    try:
        entry = json.loads(entry_path(directory, key).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(entry, dict) or entry.get("schema") != SCHEMA or entry.get("key") != key
            or entry.get("inputs") != dict(inputs)):
        return None
    try:
        check_digests(entry.get("digests"), inputs["library_sha256"], prompt_ids)
    except BaseDigestsError:
        return None
    return entry["digests"]


def store(directory: Path, key: str, inputs: Mapping[str, str], digests: dict, prompt_ids: Sequence[str]) -> None:
    """Writes `key`'s entry atomically: a temporary file in `directory`, mode ENTRY_MODE (the container writes as
    root; the host reads), flushed and fsynced, renamed over the entry. BaseDigestsError (nothing written) unless
    `digests` has check_digests' shape; OSError when the write fails (the temporary file removed)."""
    check_digests(digests, inputs["library_sha256"], prompt_ids)
    blob = json.dumps({"schema": SCHEMA, "key": key, "inputs": dict(inputs), "digests": digests}, indent=1,
                      sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=TMP_PREFIX, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), ENTRY_MODE)
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, entry_path(directory, key))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def differences(fresh: dict, cached: dict) -> list[str]:
    """The digest entries (`field` or `field[name]`) where `fresh` and `cached` differ, in DIGEST_FIELDS order."""
    out = []
    for field in DIGEST_FIELDS:
        a, b = fresh.get(field), cached.get(field)
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{field}[{k}]" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        elif a != b:
            out.append(field)
    return out


def resolve(cfg: Config, inputs: Mapping[str, str], prompt_ids: Sequence[str], compute: Callable[[], dict]) -> Result:
    """The merge-base digests by `cfg.mode` (module docstring). `inputs`: KEY_FIELDS (manifest_sha256 is
    cfg.manifest_sha256; ignored without a cache); `compute`: merge_base_digests of this run, called at most once. A
    `compute` failure propagates with nothing stored."""
    library = inputs["library_sha256"]
    key = None if cfg.directory is None else entry_key(inputs)
    if cfg.mode == "verify" and key is None:
        raise BaseDigestsError(f"{MODE_ENV}=verify needs the cache ({DIR_ENV}, {MANIFEST_ENV})")
    cached = None if key is None or cfg.mode == "fresh" else load(cfg.directory, key, inputs, prompt_ids)
    if cfg.mode == "verify" and cached is None:
        raise BaseDigestsError(f"{MODE_ENV}=verify: no valid cached entry {key} in {cfg.directory} to compare with")
    if cfg.mode == "cached" and cached is not None:
        return Result(cached, "cached", key, False, False)
    fresh = compute()
    check_digests(fresh, library, prompt_ids)
    if cfg.mode == "verify":
        differ = differences(fresh, cached)
        if differ:
            raise BaseDigestsError(f"{MODE_ENV}=verify: the fresh merge-base digests differ from cached entry {key} "
                                   f"in {', '.join(differ)}")
        return Result(fresh, "fresh", key, False, True)
    if key is None:
        return Result(fresh, "fresh", None, False, False)
    try:
        store(cfg.directory, key, inputs, fresh, prompt_ids)
    except OSError as e:
        print(f"base_digests: storing {key} in {cfg.directory} failed ({e!r}); this run's fresh digests are used",
              file=sys.stderr, flush=True)
        return Result(fresh, "fresh", key, False, False)
    return Result(fresh, "fresh", key, True, False)
