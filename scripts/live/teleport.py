#!/usr/bin/env python3
"""NetTeleport S4 (net-teleport design note 4.2 L2, 4.1, 4.3, D-T1, D-T5, D-T7): the client side of cross-install
state teleport. Python stdlib only; runs on any client device (laptop, GPU box, an agent harness's host).

The client holds the 32-byte key (D-T5) in a local key file (mode 0600, directory 0700; default
`~/.config/cere/teleport.key` or `$CERE_TELEPORT_KEY_FILE`; an OS keychain is future work) and passes it per call in
`X-Cere-State-Key` to the source's `POST /v1/cere/state/export` and the destination's `POST /v1/cere/state/import`
(docs/API.md "State teleport (NetTeleport S1)"). Both routes are on the loopback client API (F18): an endpoint URL must
be `http://127.0.0.1:<port>` (or localhost), and a remote install is reached through an ssh tunnel that this tool opens
(`--source-ssh HOST` / `--dest-ssh HOST`: the URL is then the remote host's loopback URL). Bearer tokens come from
environment variables named on the command line, never from argv.

    teleport.py keygen
    teleport.py decide --tokens 72000 --link tailnet-relay [--contended]
    teleport.py export --source URL --token-env ENV --model M --messages chat.json --dest-tier own --out s.cstb
    teleport.py export ... --to-vault node-b-wsl:/var/lib/cere/teleport-vault          # prints the vault name
    teleport.py import --dest URL --token-env ENV --model M --dest-tier own --blob s.cstb
    teleport.py import ... --from-vault node-b-wsl:/var/lib/cere/teleport-vault --name NAME
    teleport.py move --source URL --dest URL --source-token-env A --dest-token-env B --model M \\
        --messages chat.json --prompt-tokens 72000 --dest-tier own [--via direct|vault --vault SPEC] [--link NAME]
    teleport.py vault {ls,scan,rm} --vault SPEC [NAME ...]
    teleport.py probe-link --vault SPEC [--mib 256]

Tier policy (D-T7) is enforced here first: `--dest-tier` (what the destination is to this user: own/team/provider) must
be in the user's allowed tiers (`--allowed-tiers`, else `$CERE_TELEPORT_ALLOWED_TIERS`, else `own`, Q-T5), and a vault
host's `--vault-tier` too, before any byte moves; a refusal exits 2 (`refused`) and does not suggest a replay, since a
replay would send the conversation to the same disallowed tier. The source and the destination check again (S1).

Decision rule (4.2 L2): `move` ships state iff export + blob/link_rate + import + first turn after import <
re-prefill on the destination, or the destination is GPU-contended (`--contended`: a re-prefill would stall its other
decode rows, F17); otherwise, or on any refusal from either install (D-T1), it answers `"mode": "replay"` with the
reason, and the client sends its next turn to the destination, which cold-prefills (L0). The estimates are the
measured table `scripts/live/teleport_estimates.json` (written by scripts/live/teleport_bench.py), interpolated
linearly in tokens; `--link` names one of its links, `--link-mbps` gives another rate. Without an estimate (an
unmeasured board) the answer is replay.

Output is one JSON object on stdout: modes, timings, sizes, codes and the vault name. Never the key, tokens, token ids
or content (4.1). Blobs written locally (`--out`, spool files) are mode 0600 and sealed."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import secrets
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import teleport_vault as tv  # noqa: E402

TIERS = ("own", "team", "provider")
DEFAULT_ALLOWED = ("own",)
KEY_HEADER = "X-Cere-State-Key"
BLOB_TYPE = "application/vnd.cere.state-blob"
KEY_BYTES = 32
TABLE = HERE / "teleport_estimates.json"
TABLE_SCHEMA = "cere-teleport-estimates/1"
DEFAULT_BOARD = "rtx-pro-6000"
CHUNK = 1 << 20
TIMEOUT_S = 900.0
LOOPBACK = ("127.0.0.1", "localhost", "::1")
TUNNEL_READY_S = 15.0


class TeleportError(Exception):
    """A client-side failure; `code` names it (`refused` codes exit 2 without a replay)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Refused(Exception):
    """An install refused the export or the import (S1's named codes); the client replays (D-T1)."""

    def __init__(self, op: str, status: int, code: str | None) -> None:
        super().__init__(f"{op} refused: {status} {code}")
        self.op, self.status, self.code = op, status, code or f"http_{status}"


# -- key custody (D-T5) --------------------------------------------------------------------------------------------

def default_key_file(environ=os.environ) -> Path:
    return Path(environ.get("CERE_TELEPORT_KEY_FILE") or Path.home() / ".config" / "cere" / "teleport.key")


def keygen(path: Path) -> Path:
    """Creates `path` with 32 random bytes (64 hex digits), mode 0600, its directory 0700; never overwrites."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as e:
        raise TeleportError("key_file", f"{path} exists; remove it first to replace the key") from e
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(KEY_BYTES) + "\n")
    return path


def load_key(path: Path) -> str:
    """The key's 64 hex digits; refuses a file readable or writable by group or others (as ssh does)."""
    try:
        st = path.stat()
    except FileNotFoundError as e:
        raise TeleportError("key_file", f"no key file {path} (run `teleport.py keygen`)") from e
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise TeleportError("key_file", f"{path} is mode {stat.S_IMODE(st.st_mode):o}; chmod 600 it")
    text = path.read_text().strip()
    if len(text) != 2 * KEY_BYTES or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise TeleportError("key_file", f"{path} does not hold 64 hex digits")
    return text.lower()


# -- tier policy (D-T7) --------------------------------------------------------------------------------------------

def parse_tiers(text: str) -> tuple[str, ...]:
    tiers = tuple(t.strip() for t in text.split(",") if t.strip())
    bad = [t for t in tiers if t not in TIERS]
    if not tiers or bad or len(set(tiers)) != len(tiers):
        raise TeleportError("tiers", f"allowed tiers {text!r}: give distinct values of {', '.join(TIERS)}")
    return tiers


def allowed_tiers(arg: str | None, environ=os.environ) -> tuple[str, ...]:
    text = arg or environ.get("CERE_TELEPORT_ALLOWED_TIERS")
    return parse_tiers(text) if text else DEFAULT_ALLOWED


def check_tier(what: str, tier: str, allowed: Iterable[str]) -> None:
    """Refuses before any byte moves when `tier` is not one the user allowed."""
    allowed = tuple(allowed)
    if tier not in TIERS:
        raise TeleportError("tiers", f"{what} tier {tier!r} is not one of {', '.join(TIERS)}")
    if tier not in allowed:
        raise TeleportError("tier_not_allowed",
                            f"{what} tier {tier} is not in the allowed tiers [{', '.join(allowed)}]")


# -- blobs ---------------------------------------------------------------------------------------------------------

def read_header(path: Path) -> dict:
    """The JSON header of a sealed blob file (no key needed; it is the frames' AAD, nothing secret)."""
    with path.open("rb") as f:
        prefix = f.read(tv.PREFIX_BYTES)
        if len(prefix) != tv.PREFIX_BYTES or prefix[:8] != tv.MAGIC:
            raise TeleportError("blob", f"{path} is not a CERESTB1 blob")
        (length,) = struct.unpack("<I", prefix[8:12])
        if not 0 < length <= tv.MAX_HEADER_BYTES:
            raise TeleportError("blob", f"{path}: header length {length}")
        try:
            return json.loads(f.read(length))
        except ValueError as e:
            raise TeleportError("blob", f"{path}: unreadable header: {e}") from e


def private_file(path: Path) -> object:
    """`path` opened for writing, created 0600 (truncated if present)."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb")


def file_chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            yield chunk


# -- the loopback API ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Endpoint:
    """One install's client API (loopback), the caller scope's bearer and the served model."""

    url: str
    token: str
    model: str

    def connection(self, timeout: float) -> tuple[http.client.HTTPConnection, str]:
        parts = urllib.parse.urlsplit(self.url)
        if parts.scheme != "http" or parts.hostname not in LOOPBACK or parts.port is None:
            raise TeleportError("not_loopback", f"{self.url}: the state routes are on the loopback client API; "
                                                "reach a remote install with --source-ssh/--dest-ssh")
        return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout), parts.path.rstrip("/")


def error_code(raw: bytes) -> str | None:
    try:
        return (json.loads(raw).get("error") or {}).get("code")
    except (ValueError, AttributeError):
        return None


def export_body(model: str, dest_tier: str, messages: object = None, token_ids: list[int] | None = None) -> dict:
    """The export body: token ids, or a chat body (a messages list, or an object with `messages` and its other chat
    fields) rendered by the source exactly as `POST /v1/chat/completions` renders it."""
    if (messages is None) == (token_ids is None):
        raise TeleportError("usage", "give messages or token ids, not both")
    if token_ids is not None:
        body: dict = {"tokens": token_ids}
    elif isinstance(messages, list):
        body = {"messages": messages}
    elif isinstance(messages, dict) and isinstance(messages.get("messages"), list):
        body = {k: v for k, v in messages.items() if k not in ("stream", "stream_options")}
    else:
        raise TeleportError("usage", "messages must be a list or a chat body with a messages list")
    return {**body, "model": model, "destination_tier": dest_tier}


@contextlib.contextmanager
def export_stream(src: Endpoint, key: str, body: dict, timeout: float = TIMEOUT_S):
    """The source's export response, open: (meta, chunk iterator). Raises Refused on a non-200 answer."""
    conn, base = src.connection(timeout)
    try:
        t0 = time.monotonic()
        conn.request("POST", base + "/v1/cere/state/export", body=json.dumps(body).encode(),
                     headers={"Authorization": f"Bearer {src.token}", KEY_HEADER: key,
                              "Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise Refused("export", resp.status, error_code(resp.read()))
        meta = {"blob_bytes": int(resp.getheader("Content-Length") or 0),
                "tokens": int(resp.getheader("X-Cere-State-Tokens") or 0),
                "image_bytes": int(resp.getheader("X-Cere-State-Image-Bytes") or 0),
                "state_compat_id": resp.getheader("X-Cere-State-Compat-Id"),
                "headers_ms": round((time.monotonic() - t0) * 1e3, 1)}

        def chunks() -> Iterator[bytes]:
            got = 0
            while chunk := resp.read(CHUNK):
                got += len(chunk)
                yield chunk
            if got != meta["blob_bytes"]:
                raise TeleportError("transfer_failed", f"export stream ended at {got} of {meta['blob_bytes']} bytes")

        yield meta, chunks()
    except (OSError, http.client.HTTPException) as e:
        raise TeleportError("transfer_failed", f"export from {src.url}: {e}") from e
    finally:
        conn.close()


def export_to_file(src: Endpoint, key: str, body: dict, out: Path, timeout: float = TIMEOUT_S) -> dict:
    """Export into the sealed file `out` (0600). {blob_bytes, tokens, image_bytes, state_compat_id, ms}."""
    t0 = time.monotonic()
    try:
        with export_stream(src, key, body, timeout) as (meta, chunks), private_file(out) as f:
            for chunk in chunks:
                f.write(chunk)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return {**meta, "ms": round((time.monotonic() - t0) * 1e3, 1)}


def import_chunks(dst: Endpoint, key: str, chunks: Iterable[bytes], length: int,
                  timeout: float = TIMEOUT_S) -> dict:
    """Streams a blob of `length` bytes into the destination's import. {imported_tokens, image_bytes, tier, ms}."""
    conn, base = dst.connection(timeout)
    t0 = time.monotonic()
    try:
        target = base + "/v1/cere/state/import?model=" + urllib.parse.quote(dst.model, safe="")
        try:
            conn.request("POST", target, body=iter(chunks),
                         headers={"Authorization": f"Bearer {dst.token}", KEY_HEADER: key,
                                  "Content-Type": BLOB_TYPE, "Content-Length": str(length)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the destination refused before the body ended; its answer follows
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise Refused("import", resp.status, error_code(raw))
        answer = json.loads(raw)
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise TeleportError("transfer_failed", f"import into {dst.url}: {e}") from e
    finally:
        conn.close()
    return {"imported_tokens": answer.get("imported_tokens"), "image_bytes": answer.get("image_bytes"),
            "tier": answer.get("tier"), "ms": round((time.monotonic() - t0) * 1e3, 1)}


def import_file(dst: Endpoint, key: str, path: Path, timeout: float = TIMEOUT_S) -> dict:
    read_header(path)
    return import_chunks(dst, key, file_chunks(path), path.stat().st_size, timeout)


def move_direct(src: Endpoint, dst: Endpoint, key: str, body: dict, timeout: float = TIMEOUT_S) -> dict:
    """Source -> destination with no copy at rest: the export response is the import request body."""
    t0 = time.monotonic()
    with export_stream(src, key, body, timeout) as (meta, chunks):
        imported = import_chunks(dst, key, chunks, meta["blob_bytes"], timeout)
    return {"export": meta, "import": imported, "wall_ms": round((time.monotonic() - t0) * 1e3, 1)}


# -- ssh tunnels (F18, R-T10) --------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tunnel_argv(host: str, local_port: int, remote_port: int) -> list[str]:
    return ["ssh", "-N", *tv.SSH_OPTS, "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", host]


@contextlib.contextmanager
def tunnel(url: str, host: str | None, popen: Callable[..., subprocess.Popen] = subprocess.Popen):
    """`url` itself without `host`; else a local URL forwarded over `ssh -L` to `url`'s port on `host`'s loopback."""
    if host is None:
        yield url
        return
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in LOOPBACK or parts.port is None:
        raise TeleportError("not_loopback", f"{url}: give the remote install's loopback URL with --*-ssh")
    local = free_port()
    proc = popen(tunnel_argv(host, local, parts.port), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + TUNNEL_READY_S
        while True:
            if proc.poll() is not None:
                raise TeleportError("transfer_failed", f"ssh tunnel to {host} exited {proc.returncode}")
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", local), timeout=1):
                break
            if time.monotonic() > deadline:
                raise TeleportError("transfer_failed", f"ssh tunnel to {host} not up in {TUNNEL_READY_S} s")
            time.sleep(0.1)
        yield urllib.parse.urlunsplit(("http", f"127.0.0.1:{local}", parts.path, parts.query, ""))
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(5)


# -- the decision rule (4.2 L2) ------------------------------------------------------------------------------------

ROW_FIELDS = ("tokens", "blob_bytes", "export_ms", "import_ms", "first_turn_ms", "reprefill_ms")


def load_table(path: Path = TABLE) -> dict:
    table = json.loads(path.read_text())
    if table.get("schema") != TABLE_SCHEMA:
        raise TeleportError("table", f"{path}: schema is not {TABLE_SCHEMA}")
    for board, entry in table.get("boards", {}).items():
        rows = entry.get("rows") or []
        if len(rows) < 2 or any(f not in r for r in rows for f in ROW_FIELDS):
            raise TeleportError("table", f"{path}: board {board} needs >= 2 rows with {', '.join(ROW_FIELDS)}")
    return table


def interpolate(rows: list[dict], field: str, tokens: float) -> float:
    """Piecewise linear in tokens through the measured rows, extended along the outer segments, never below 0."""
    pts = sorted((r["tokens"], r[field]) for r in rows)
    if tokens <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif tokens >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        i = next(i for i in range(1, len(pts)) if tokens <= pts[i][0])
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    y = y0 if x1 == x0 else y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return max(y, 0.0)


def link_rate(table: dict, link: str | None, mbps: float | None) -> tuple[str, float | None]:
    """(name, effective MB/s of the extra transfer; None: no hop beyond export and import, as on one host)."""
    if mbps is not None:
        if mbps <= 0:
            raise TeleportError("usage", "--link-mbps must be > 0")
        return f"{mbps} MB/s", mbps
    name = link or "loopback"
    links = table.get("links") or {}
    if name not in links:
        raise TeleportError("usage", f"link {name!r} is not in the table ({', '.join(sorted(links))})")
    return name, links[name].get("transfer_mb_per_s")


def decide(table: dict, tokens: int, *, board: str = DEFAULT_BOARD, link: str | None = None,
           link_mbps: float | None = None, contended: bool = False) -> dict:
    """{"mode": "state"|"replay", "reason", estimates}: ship iff export + transfer + import + first turn <
    re-prefill on the destination, or the destination is GPU-contended."""
    name, rate = link_rate(table, link, link_mbps)
    entry = (table.get("boards") or {}).get(board)
    if entry is None:
        return {"mode": "replay", "reason": "no_estimate_for_board", "board": board, "link": name}
    rows = entry["rows"]
    est = {f: interpolate(rows, f, tokens) for f in ROW_FIELDS if f != "tokens"}
    transfer_ms = 0.0 if rate is None else est["blob_bytes"] / (rate * 1e6) * 1e3
    teleport_ms = est["export_ms"] + transfer_ms + est["import_ms"] + est["first_turn_ms"]
    out = {"board": board, "link": name, "tokens": tokens, "blob_bytes": round(est["blob_bytes"]),
           "transfer_ms": round(transfer_ms, 1), "teleport_ms": round(teleport_ms, 1),
           "reprefill_ms": round(est["reprefill_ms"], 1)}
    if contended:
        return {"mode": "state", "reason": "destination_contended", **out}
    if teleport_ms < est["reprefill_ms"]:
        return {"mode": "state", "reason": "state_faster", **out}
    return {"mode": "replay", "reason": "reprefill_faster", **out}


# -- the commands --------------------------------------------------------------------------------------------------

def token(environ, name: str) -> str:
    value = environ.get(name)
    if not value:
        raise TeleportError("usage", f"${name} (the bearer token) is not set")
    return value


def payload(args, dest_tier: str) -> tuple[dict, int | None]:
    """(export body, its token count when known) from --messages / --token-ids."""
    if bool(args.messages) == bool(args.token_ids):
        raise TeleportError("usage", "give --messages FILE or --token-ids FILE")
    if args.token_ids:
        ids = json.loads(Path(args.token_ids).read_text())
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and 0 <= i < 2**32 for i in ids):
            raise TeleportError("usage", f"{args.token_ids}: not a non-empty JSON list of token ids")
        return export_body(args.model, dest_tier, token_ids=ids), len(ids)
    return export_body(args.model, dest_tier, messages=json.loads(Path(args.messages).read_text())), None


def spool_dir() -> Path:
    path = Path(os.environ.get("CERE_TELEPORT_SPOOL") or Path(tempfile.gettempdir()) / f"cere-teleport-{os.getuid()}")
    path.mkdir(mode=0o700, exist_ok=True)
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise TeleportError("spool", f"{path} is not private (mode 0700)")
    return path


def replay(reason: str, **extra) -> dict:
    return {"mode": "replay", "reason": reason, **extra}


def cmd_export(args, environ) -> dict:
    allowed = allowed_tiers(args.allowed_tiers, environ)
    check_tier("destination", args.dest_tier, allowed)
    vault = tv.Vault.parse(args.to_vault) if args.to_vault else None
    if vault is not None:
        check_tier("vault", args.vault_tier, allowed)
    if (args.out is None) == (vault is None):
        raise TeleportError("usage", "give --out FILE or --to-vault SPEC")
    key = load_key(Path(args.key_file) if args.key_file else default_key_file(environ))
    body, _ = payload(args, args.dest_tier)
    with tunnel(args.source, args.source_ssh) as url:
        src = Endpoint(url, token(environ, args.token_env), args.model)
        out = Path(args.out) if args.out else spool_dir() / f"{tv.new_name()}{tv.SUFFIX}"
        result = {"export": export_to_file(src, key, body, out)}
    if vault is None:
        return {**result, "out": str(out)}
    try:
        name = args.name or tv.new_name()
        result["vault_put"] = vault.put(out, name)
        return {**result, "vault": str(vault), "name": name}
    finally:
        out.unlink(missing_ok=True)


def cmd_import(args, environ) -> dict:
    check_tier("destination", args.dest_tier, allowed_tiers(args.allowed_tiers, environ))
    key = load_key(Path(args.key_file) if args.key_file else default_key_file(environ))
    vault = tv.Vault.parse(args.from_vault) if args.from_vault else None
    if (args.blob is None) == (vault is None) or (vault is not None and not args.name):
        raise TeleportError("usage", "give --blob FILE or --from-vault SPEC --name NAME")
    result: dict = {}
    path = Path(args.blob) if args.blob else spool_dir() / f"{tv.check_name(args.name)}{tv.SUFFIX}"
    try:
        if vault is not None:
            result["vault_get"] = vault.get(args.name, path)
        with tunnel(args.dest, args.dest_ssh) as url:
            result["import"] = import_file(Endpoint(url, token(environ, args.token_env), args.model), key, path)
        if vault is not None and not args.keep:
            vault.remove([args.name])
    finally:
        if vault is not None:
            path.unlink(missing_ok=True)
    return result


def cmd_move(args, environ) -> dict:
    allowed = allowed_tiers(args.allowed_tiers, environ)
    check_tier("destination", args.dest_tier, allowed)
    vault = tv.Vault.parse(args.vault) if args.vault else None
    if args.via == "vault":
        if vault is None:
            raise TeleportError("usage", "--via vault needs --vault SPEC")
        check_tier("vault", args.vault_tier, allowed)
    body, known = payload(args, args.dest_tier)
    tokens = known if known is not None else args.prompt_tokens
    decision = None
    if not args.force_state:
        if tokens is None:
            raise TeleportError("usage", "the decision needs --prompt-tokens N (or --token-ids, or --force-state)")
        link = args.link or ("tailnet-relay" if args.via == "vault" else "loopback")
        decision = decide(load_table(Path(args.table)), tokens, board=args.board,
                          link=None if args.link_mbps else link, link_mbps=args.link_mbps,
                          contended=args.contended)
        if decision["mode"] == "replay":
            return {**replay(decision["reason"]), "decision": decision}
    key = load_key(Path(args.key_file) if args.key_file else default_key_file(environ))
    t0 = time.monotonic()
    try:
        with tunnel(args.source, args.source_ssh) as src_url, tunnel(args.dest, args.dest_ssh) as dst_url:
            src = Endpoint(src_url, token(environ, args.source_token_env), args.model)
            dst = Endpoint(dst_url, token(environ, args.dest_token_env), args.model)
            if args.via == "direct":
                moved = move_direct(src, dst, key, body)
            else:
                moved = vault_move(src, dst, key, body, vault)
    except Refused as e:
        return {**replay(e.code, refused_by=e.op), "decision": decision}
    except (TeleportError, tv.VaultError) as e:
        return {**replay("transfer_failed", detail=str(e)[:300]), "decision": decision}
    return {"mode": "state", "reason": "imported", "decision": decision, **moved,
            "wall_ms": round((time.monotonic() - t0) * 1e3, 1)}


def vault_move(src: Endpoint, dst: Endpoint, key: str, body: dict, vault: tv.Vault) -> dict:
    """Export -> spool -> vault -> spool -> import; the local and vault copies are removed afterwards."""
    spool = spool_dir()
    name = tv.new_name()
    sent, got = spool / f"{name}.out{tv.SUFFIX}", spool / f"{name}.in{tv.SUFFIX}"
    try:
        exported = export_to_file(src, key, body, sent)
        put = vault.put(sent, name)
        sent.unlink()
        fetched = vault.get(name, got)
        imported = import_file(dst, key, got)
        vault.remove([name])
        return {"export": exported, "vault_put": put, "vault_get": fetched, "import": imported}
    finally:
        sent.unlink(missing_ok=True)
        got.unlink(missing_ok=True)


def cmd_vault(args, environ) -> object:
    vault = tv.Vault.parse(args.vault)
    if args.op == "ls":
        return vault.list()
    if args.op == "scan":
        return vault.scan(args.names or None)
    if not args.names:
        raise TeleportError("usage", "vault rm needs NAME ...")
    return {"removed": vault.remove(args.names)}


def cmd_probe_link(args, environ) -> dict:
    """Push and pull `--mib` MiB of random bytes through the vault; the link rates the decision rule reads."""
    vault = tv.Vault.parse(args.vault)
    spool = spool_dir()
    name = tv.new_name()
    sent, got = spool / f"{name}.probe", spool / f"{name}.probe.back"
    try:
        with private_file(sent) as f:
            for _ in range(args.mib):
                f.write(os.urandom(CHUNK))
        put = vault.put(sent, name)
        fetched = vault.get(name, got)
        vault.remove([name])
    finally:
        sent.unlink(missing_ok=True)
        got.unlink(missing_ok=True)
    size = args.mib * CHUNK
    push_ms, pull_ms = max(put["ms"], 0.1), max(fetched["ms"], 0.1)
    return {"bytes": size, "push_ms": put["ms"], "pull_ms": fetched["ms"],
            "push_mb_per_s": round(size / push_ms / 1e3, 1), "pull_mb_per_s": round(size / pull_ms / 1e3, 1),
            "relay_mb_per_s": round(size / (push_ms + pull_ms) / 1e3, 1)}


def cmd_decide(args, environ) -> dict:
    return decide(load_table(Path(args.table)), args.tokens, board=args.board, link=args.link,
                  link_mbps=args.link_mbps, contended=args.contended)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key-file", help="the 32-byte key file (default $CERE_TELEPORT_KEY_FILE or "
                                      "~/.config/cere/teleport.key)")
    sub = p.add_subparsers(dest="command", required=True)

    def rule(q) -> None:
        q.add_argument("--table", default=str(TABLE), help="the measured estimates table")
        q.add_argument("--board", default=DEFAULT_BOARD, help="the destination board in the table")
        q.add_argument("--link", help="a link of the table: loopback (one host), tailnet-hop (one hop between two "
                                      "hosts), tailnet-relay (through a vault; the --via vault default)")
        q.add_argument("--link-mbps", type=float, help="the transfer rate in MB/s instead of a table link")
        q.add_argument("--contended", action="store_true",
                       help="the destination GPU serves other rows: a re-prefill would stall them, ship state")

    def tiers(q) -> None:
        q.add_argument("--dest-tier", required=True, choices=TIERS, help="what the destination is to you")
        q.add_argument("--allowed-tiers", help="comma list (default $CERE_TELEPORT_ALLOWED_TIERS or own)")

    def source(q) -> None:
        q.add_argument("--model", required=True)
        q.add_argument("--messages", help="JSON: a messages list or a chat body")
        q.add_argument("--token-ids", help="JSON list of the prefix's token ids")

    sub.add_parser("keygen", help="create the key file (never overwrites)")
    q = sub.add_parser("decide", help="the decision rule alone")
    q.add_argument("--tokens", type=int, required=True)
    rule(q)
    q = sub.add_parser("export", help="seal a cached prefix from the source into a file or a vault")
    q.add_argument("--source", required=True, help="the source's loopback URL")
    q.add_argument("--source-ssh", help="reach --source over an ssh tunnel to this host")
    q.add_argument("--token-env", required=True, help="env var holding the source scope's bearer")
    source(q)
    tiers(q)
    q.add_argument("--out", help="the sealed blob file (written 0600)")
    q.add_argument("--to-vault", help="host:/abs/dir or /abs/dir")
    q.add_argument("--vault-tier", default="own", choices=TIERS)
    q.add_argument("--name", help="the vault name (default random)")
    q = sub.add_parser("import", help="import a sealed blob into the destination")
    q.add_argument("--dest", required=True)
    q.add_argument("--dest-ssh")
    q.add_argument("--token-env", required=True)
    q.add_argument("--model", required=True)
    tiers(q)
    q.add_argument("--blob")
    q.add_argument("--from-vault")
    q.add_argument("--name")
    q.add_argument("--keep", action="store_true", help="keep the vault copy after the import")
    q = sub.add_parser("move", help="decide, then export -> transfer -> import, or answer replay")
    q.add_argument("--source", required=True)
    q.add_argument("--source-ssh")
    q.add_argument("--dest", required=True)
    q.add_argument("--dest-ssh")
    q.add_argument("--source-token-env", required=True)
    q.add_argument("--dest-token-env", required=True)
    source(q)
    tiers(q)
    q.add_argument("--prompt-tokens", type=int, help="the prefix length the decision prices (e.g. last usage)")
    q.add_argument("--via", choices=("direct", "vault"), default="direct")
    q.add_argument("--vault")
    q.add_argument("--vault-tier", default="own", choices=TIERS)
    q.add_argument("--force-state", action="store_true", help="skip the decision rule (measurement)")
    rule(q)
    q = sub.add_parser("vault", help="list, scan (at-rest check) or remove vault blobs")
    q.add_argument("op", choices=("ls", "scan", "rm"))
    q.add_argument("--vault", required=True)
    q.add_argument("names", nargs="*")
    q = sub.add_parser("probe-link", help="measure push/pull MB/s through a vault with random bytes")
    q.add_argument("--vault", required=True)
    q.add_argument("--mib", type=int, default=256)
    return p


COMMANDS = {"decide": cmd_decide, "export": cmd_export, "import": cmd_import, "move": cmd_move,
            "vault": cmd_vault, "probe-link": cmd_probe_link}


def main(argv: list[str] | None = None, environ=os.environ) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "keygen":
            result: object = {"key_file": str(keygen(Path(args.key_file) if args.key_file
                                                       else default_key_file(environ)))}
        else:
            result = COMMANDS[args.command](args, environ)
    except Refused as e:
        print(json.dumps({**replay(e.code, refused_by=e.op), "status": e.status}, sort_keys=True))
        return 1
    except TeleportError as e:
        print(json.dumps({"mode": "refused" if e.code == "tier_not_allowed" else "error", "code": e.code,
                          "detail": str(e)}, sort_keys=True))
        return 2
    except tv.VaultError as e:
        print(json.dumps({"mode": "error", "code": "vault", "detail": str(e)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
