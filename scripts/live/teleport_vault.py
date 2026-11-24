#!/usr/bin/env python3
"""NetTeleport S4 (net-teleport design note 4.2 L2, 4.1, 4.3, D-T5): the sealed-blob vault. Python stdlib only.

A vault is a directory that holds `CERESTB1` blobs between an export and an import when the source and the destination
are not online together: on the client's own disk (`/abs/dir`) or on another host reached over ssh/scp
(`host:/abs/dir`, e.g. node-b over the tailnet, `node-b-wsl:/var/lib/cere/teleport-vault`). The vault never sees the key:
it stores, lists, scans and deletes sealed files only, named by a random id (never by content or session).

`scan` proves at rest what the vault holds without the key (bar S4 "no plaintext image on the vault's disk"): each file
must be a format-1 `CERESTB1` blob whose size matches its header, with no `CERESNP1` magic anywhere, no run of 16
consecutive little-endian u32 values below 2^18 (a token-id run of image kind 0; the vocabulary is 248,320 ids), and
at least `ENTROPY_FLOOR` bits per byte in every sampled 1 MiB window of the frames (AES-GCM ciphertext measures about
7.9998). On a remote vault this file runs there as `python3 - --scan DIR NAME...` (its source on stdin; the remote
needs python3 >= 3.8 and nothing else), so the bytes are read where they rest and only the verdict travels back.

    python3 scripts/live/teleport_vault.py --scan /abs/dir [NAME ...]     # what the remote side runs
    python3 scripts/live/teleport_vault.py --list /abs/dir
    python3 scripts/live/teleport_vault.py --rm /abs/dir NAME [NAME ...]

The client side (put/get over scp, the remote calls above) is `Vault`; `scripts/live/teleport.py vault ...` is its CLI.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shlex
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MAGIC = b"CERESTB1"
IMAGE_MAGIC = b"CERESNP1"
FORMAT = 1
PREFIX_BYTES = 12
MAX_HEADER_BYTES = 4096
TAG_BYTES = 16
SUFFIX = ".cstb"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ROOT_RE = re.compile(r"^/[\w./-]+$", re.ASCII)
# A token-id run of image kind 0: 16 consecutive u32 below 2^18 (top byte 0, third byte <= 3). In uniform bytes the
# chance per offset is 2^-224; the literal zero first lets `re` skip ahead (several GB/s).
TOKEN_RUN = re.compile(rb"\x00(?:[\x00-\xff]{2}[\x00-\x03]\x00){15}")
TOKEN_RUN_BYTES = 61
SCAN_CHUNK = 16 << 20
ENTROPY_WINDOW = 1 << 20
ENTROPY_FLOOR = 7.99
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
HERE = Path(__file__).resolve()

Runner = Callable[..., subprocess.CompletedProcess]


class VaultError(Exception):
    """A vault operation failed (transport, path or name)."""


def entropy(window: bytes) -> float:
    """Shannon entropy of `window` in bits per byte."""
    n = len(window)
    return -sum(c / n * math.log2(c / n) for c in Counter(window).values()) if n else 0.0


def scan_file(path: str | Path) -> dict:
    """The at-rest verdict of one vault file: shape against its header, image magic, token-id runs, entropy."""
    path = Path(path)
    size = path.stat().st_size
    out: dict = {"bytes": size, "magic": False, "header": False, "size_matches": False, "image_magic": 0,
                 "token_id_runs": 0, "entropy_min": None, "entropy_windows": 0}
    frames_at = size
    with path.open("rb") as f:
        prefix = f.read(PREFIX_BYTES)
        out["magic"] = len(prefix) == PREFIX_BYTES and prefix[:8] == MAGIC
        if out["magic"]:
            (length,) = struct.unpack("<I", prefix[8:12])
            if 0 < length <= MAX_HEADER_BYTES:
                try:
                    header = json.loads(f.read(length))
                except ValueError:
                    header = None
                if isinstance(header, dict) and header.get("format") == FORMAT:
                    out["header"] = True
                    frames_at = PREFIX_BYTES + length
                    image = int(header.get("image_bytes") or 0)
                    frame = int(header.get("frame_bytes") or 0)
                    if frame > 0:
                        tags = -(-image // frame) * TAG_BYTES
                        out["size_matches"] = size == frames_at + image + tags
                    out["tokens"] = header.get("tokens")
        f.seek(0)
        carry, offset, low = b"", 0, None
        while True:
            chunk = f.read(SCAN_CHUNK)
            if not chunk:
                break
            buf = carry + chunk
            # A match wholly inside `carry` was counted in the previous pass.
            out["image_magic"] += sum(1 for m in re.finditer(re.escape(IMAGE_MAGIC), buf) if m.end() > len(carry))
            out["token_id_runs"] += sum(1 for m in TOKEN_RUN.finditer(buf) if m.end() > len(carry))
            start = max(frames_at - offset, 0)
            if start < len(chunk):
                window = chunk[start:start + ENTROPY_WINDOW]
                if len(window) >= 4096:
                    e = entropy(window)
                    low = e if low is None else min(low, e)
                    out["entropy_windows"] += 1
            offset += len(chunk)
            carry = buf[-(TOKEN_RUN_BYTES - 1):]
    out["entropy_min"] = None if low is None else round(low, 5)
    out["sealed"] = (out["magic"] and out["header"] and out["size_matches"] and out["image_magic"] == 0
                     and out["token_id_runs"] == 0 and low is not None and low >= ENTROPY_FLOOR)
    return out


def blob_names(root: str | Path) -> list[str]:
    return sorted(p.name[:-len(SUFFIX)] for p in Path(root).iterdir()
                  if p.is_file() and p.name.endswith(SUFFIX) and NAME_RE.match(p.name[:-len(SUFFIX)]))


def scan_dir(root: str | Path, names: list[str] | None = None) -> dict:
    """Every named blob (default: all) plus any other file in `root`, each scanned; `pass` iff all are sealed."""
    root = Path(root)
    files = sorted(p for p in root.iterdir() if p.is_file()) if root.is_dir() else []
    wanted = None if not names else {n + SUFFIX for n in names}
    rows = {p.name: scan_file(p) for p in files if wanted is None or p.name in wanted}
    missing = sorted(wanted - set(rows)) if wanted else []
    return {"dir": str(root), "files": rows, "missing": missing,
            "pass": bool(rows) and not missing and all(r["sealed"] for r in rows.values())}


def list_dir(root: str | Path) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    return [{"name": n, "bytes": (root / (n + SUFFIX)).stat().st_size,
             "mtime_unix": int((root / (n + SUFFIX)).stat().st_mtime)} for n in blob_names(root)]


def remove(root: str | Path, names: list[str]) -> list[str]:
    removed = []
    for n in names:
        check_name(n)
        p = Path(root) / (n + SUFFIX)
        if p.exists():
            p.unlink()
            removed.append(n)
    return removed


def check_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise VaultError(f"vault name {name!r} is not [A-Za-z0-9][A-Za-z0-9_-]{{0,63}}")
    return name


def new_name() -> str:
    """A random vault id: never derived from the session, the prompt or the blob."""
    return secrets.token_hex(16)


@dataclass(frozen=True)
class Vault:
    """`host:/abs/dir` (ssh/scp; `host` is an ssh destination) or `/abs/dir` (this machine)."""

    host: str | None
    root: str

    @staticmethod
    def parse(spec: str) -> "Vault":
        host, sep, root = spec.rpartition(":") if ":" in spec else ("", "", spec)
        if sep and not host:
            raise VaultError(f"vault {spec!r}: empty host")
        if not ROOT_RE.match(root) or "/../" in root + "/" or root.endswith("/.."):
            raise VaultError(f"vault {spec!r}: the directory must be absolute ([A-Za-z0-9_./-] only)")
        return Vault(host or None, root.rstrip("/") or "/")

    def __str__(self) -> str:
        return f"{self.host}:{self.root}" if self.host else self.root

    def path(self, name: str) -> str:
        return f"{self.root}/{check_name(name)}{SUFFIX}"

    def _ssh(self, run: Runner, command: str, stdin: bytes | None = None) -> bytes:
        proc = run(["ssh", *SSH_OPTS, self.host, command], input=stdin, capture_output=True)
        if proc.returncode != 0:
            raise VaultError(f"ssh {self.host} failed ({proc.returncode}): {_text(proc.stderr)[-300:]}")
        return proc.stdout

    def _module(self, run: Runner, *args: str) -> object:
        """This file run on the vault host (`python3 - <args>`, source on stdin); its JSON answer."""
        command = "python3 - " + " ".join(shlex.quote(a) for a in args)
        out = self._ssh(run, command, HERE.read_bytes())
        try:
            return json.loads(out)
        except ValueError as e:
            raise VaultError(f"{self.host}: unreadable answer from the vault helper: {e}") from e

    def put(self, local: str | Path, name: str, run: Runner = subprocess.run) -> dict:
        """Uploads the sealed file `local` as `name` (written as a dot part, mode 0600, then renamed)."""
        target = self.path(name)
        part = f"{self.root}/.{name}{SUFFIX}.part"
        t0 = time.monotonic()
        if self.host is None:
            os.makedirs(self.root, mode=0o700, exist_ok=True)
            _copy(Path(local), Path(part))
            os.replace(part, target)
        else:
            self._ssh(run, f"umask 077 && mkdir -p -- {shlex.quote(self.root)}")
            proc = run(["scp", "-q", *SSH_OPTS, str(local), f"{self.host}:{part}"], capture_output=True)
            if proc.returncode != 0:
                raise VaultError(f"scp to {self.host} failed ({proc.returncode}): {_text(proc.stderr)[-300:]}")
            self._ssh(run, f"chmod 600 -- {shlex.quote(part)} && mv -f -- {shlex.quote(part)} {shlex.quote(target)}")
        return {"name": name, "bytes": Path(local).stat().st_size, "ms": round((time.monotonic() - t0) * 1e3, 1)}

    def get(self, name: str, local: str | Path, run: Runner = subprocess.run) -> dict:
        """Downloads `name` to `local` (mode 0600; a part file renamed once complete)."""
        source, local = self.path(name), Path(local)
        part = local.with_name(f".{local.name}.part")
        t0 = time.monotonic()
        if self.host is None:
            _copy(Path(source), part)
        else:
            proc = run(["scp", "-q", *SSH_OPTS, f"{self.host}:{source}", str(part)], capture_output=True)
            if proc.returncode != 0:
                part.unlink(missing_ok=True)
                raise VaultError(f"scp from {self.host} failed ({proc.returncode}): {_text(proc.stderr)[-300:]}")
            os.chmod(part, 0o600)
        os.replace(part, local)
        return {"name": name, "bytes": local.stat().st_size, "ms": round((time.monotonic() - t0) * 1e3, 1)}

    def scan(self, names: list[str] | None = None, run: Runner = subprocess.run) -> dict:
        for n in names or ():
            check_name(n)
        if self.host is None:
            return scan_dir(self.root, names)
        return self._module(run, "--scan", self.root, *(names or ()))

    def list(self, run: Runner = subprocess.run) -> list[dict]:
        return list_dir(self.root) if self.host is None else self._module(run, "--list", self.root)

    def remove(self, names: list[str], run: Runner = subprocess.run) -> list[str]:
        for n in names:
            check_name(n)
        return remove(self.root, names) if self.host is None else self._module(run, "--rm", self.root, *names)

    def remove_dir(self, run: Runner = subprocess.run) -> None:
        """Removes every blob then the (then empty) directory; refuses a directory with other files."""
        self.remove([e["name"] for e in self.list(run)], run)
        if self.host is None:
            os.rmdir(self.root)
        else:
            self._ssh(run, f"rmdir -- {shlex.quote(self.root)}")


def _copy(src: Path, dst: Path) -> None:
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with src.open("rb") as r, os.fdopen(fd, "wb") as w:
        while chunk := r.read(SCAN_CHUNK):
            w.write(chunk)


def _text(raw: bytes | str | None) -> str:
    return raw.decode(errors="replace") if isinstance(raw, bytes) else (raw or "")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[0] not in ("--scan", "--list", "--rm") or not ROOT_RE.match(args[1]):
        print(__doc__, file=sys.stderr)
        return 2
    op, root, names = args[0], args[1], args[2:]
    try:
        for n in names:
            check_name(n)
        result = scan_dir(root, names) if op == "--scan" else list_dir(root) if op == "--list" \
            else remove(root, names)
    except VaultError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
