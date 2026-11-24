"""Reference NEXTN token-map converter (M2.1 Q10): torch stored-zip `*/data.pkl` int list -> little-endian u32.

No unpickling: a fixed zip reader and a pickle grammar. CRC is not checked (the source sha256 is recorded instead).
`python3 scripts/token_map.py PT --config CONFIG_JSON [--out FILE]` writes the u32le bytes (stdout by default).
Rejections exit 2 with `<error_class>: <message>` as the first stderr line.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from cere_contracts import ArtifactError, limits_with, parse_json

EOCD_SIG, CD_SIG, LOCAL_SIG = 0x06054B50, 0x02014B50, 0x04034B50
ZIP64_RECORD_SIG, ZIP64_LOCATOR_SIG = 0x06064B50, 0x07064B50
TRANSFORM = "pickle_int_list_to_u32le:v1"
ZIP_SLACK_BYTES = 1 << 20  # zip structure and other entries a token-map file may carry beyond its data.pkl limit


def max_file_bytes(limits: dict) -> int:
    """The largest token-map file read whole (cerec's `token_map::max_file_bytes`)."""
    return limits["max_token_map_pkl_bytes"] + ZIP_SLACK_BYTES


def _u(fmt: str, data: bytes, off: int, cls: str = "zip_structure"):
    size = struct.calcsize(fmt)
    if off < 0 or off + size > len(data):
        raise ArtifactError(cls, f"truncated record at {off}")
    return struct.unpack_from(fmt, data, off)


def _zip64_trailer(data: bytes, eocd: int, classic: tuple) -> int:
    """Where the central directory must end: the EOCD, or PyTorch's zip64 end record when a locator sits right
    before the EOCD. The record + locator are accepted only when exactly consistent with the classic EOCD."""
    loc = eocd - 20
    if loc < 0 or data[loc:loc + 4] != struct.pack("<I", ZIP64_LOCATOR_SIG):
        return eocd
    _, loc_disk, rec, total = _u("<IIQI", data, loc)
    if loc_disk != 0 or total != 1:
        raise ArtifactError("zip_structure", "zip64 locator disk fields")
    if rec + 56 != loc:
        raise ArtifactError("zip_structure", "zip64 locator does not point at a record ending at the locator")
    sig, size, _made, _need, *fields = _u("<IQHHIIQQQQ", data, rec)
    if sig != ZIP64_RECORD_SIG or size != 44:
        raise ArtifactError("zip_structure", "zip64 end record signature or size")
    if tuple(fields) != classic:
        raise ArtifactError("zip_structure", "zip64 end record disagrees with the end-of-central-directory")
    return rec


def find_data_pkl(data: bytes, limits: dict) -> bytes:
    """Return the stored bytes of the single `<archive>/data.pkl` entry."""
    if len(data) > max_file_bytes(limits):
        raise ArtifactError("zip_entry", f"token-map file is {len(data)} bytes > {max_file_bytes(limits)}")
    eocd = None
    for i in range(len(data) - 22, max(-1, len(data) - 65557 - 1), -1):
        if (data[i:i + 4] == struct.pack("<I", EOCD_SIG) and
                i + 22 + struct.unpack_from("<H", data, i + 20)[0] == len(data)):
            eocd = i
            break
    if eocd is None:
        raise ArtifactError("zip_structure", "no end-of-central-directory record")
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_off, _ = _u("<IHHHHIIH", data, eocd)
    if disk or cd_disk or n_disk != n_total:
        raise ArtifactError("zip_unsupported", "multi-disk archive")
    if 0xFFFF in (disk, cd_disk, n_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_off):
        raise ArtifactError("zip_unsupported", "zip64 archive")
    cd_limit = _zip64_trailer(data, eocd, (disk, cd_disk, n_disk, n_total, cd_size, cd_off))
    if cd_off + cd_size > cd_limit:
        raise ArtifactError("zip_structure", "central directory out of range")
    if struct.pack("<I", ZIP64_LOCATOR_SIG) in data[cd_off + cd_size:cd_limit]:
        raise ArtifactError("zip_structure", "zip64 locator not immediately before the end-of-central-directory")
    pos, end = cd_off, cd_off + cd_size
    entries = []
    for _ in range(n_total):
        (sig, _vm, _vn, flags, method, _t, _d, _crc, csize, usize, nlen, xlen, clen, disk_start, _ia, _ea,
         local_off) = _u("<IHHHHHHIIIHHHHHII", data, pos)
        if sig != CD_SIG or pos + 46 + nlen + xlen + clen > end:
            raise ArtifactError("zip_structure", f"bad central directory entry at {pos}")
        name = data[pos + 46:pos + 46 + nlen]
        if flags & 1:
            raise ArtifactError("zip_unsupported", f"encrypted entry {name!r}")
        if method != 0:
            raise ArtifactError("zip_unsupported", f"entry {name!r} uses compression method {method}")
        if disk_start:
            raise ArtifactError("zip_unsupported", f"entry {name!r} starts on disk {disk_start}")
        if 0xFFFFFFFF in (csize, usize, local_off):
            raise ArtifactError("zip_unsupported", f"zip64 entry {name!r}")
        entries.append((name, csize, usize, local_off))
        pos += 46 + nlen + xlen + clen
    if pos != end:
        raise ArtifactError("zip_structure", "central directory size mismatch")
    hits = [e for e in entries if e[0].endswith(b"/data.pkl") and e[0].count(b"/") == 1 and len(e[0]) > 9]
    if len(hits) != 1:
        raise ArtifactError("zip_entry", f"{len(hits)} */data.pkl entries")
    name, csize, usize, local_off = hits[0]
    if csize != usize or usize > limits["max_token_map_pkl_bytes"]:
        raise ArtifactError("zip_entry", f"data.pkl sizes {csize}/{usize} invalid or over limit")
    if local_off + 30 > cd_off:
        raise ArtifactError("zip_entry", "data.pkl local header out of range")
    sig, *_rest, nlen, xlen = _u("<IHHHHHIIIHH", data, local_off, "zip_entry")
    if sig != LOCAL_SIG or data[local_off + 30:local_off + 30 + nlen] != name:
        raise ArtifactError("zip_entry", "data.pkl local header signature or name mismatch")
    start = local_off + 30 + nlen + xlen
    if start + csize > cd_off:
        raise ArtifactError("zip_entry", "data.pkl data runs into the central directory")
    return data[start:start + csize]


def parse_int_list(pkl: bytes) -> list[int]:
    """Q10 grammar: PROTO 2..5, EMPTY_LIST, [BINPUT], {MARK int* APPENDS | int APPEND}*, STOP (last byte)."""
    p = 0

    def take(n):
        nonlocal p
        if p + n > len(pkl):
            raise ArtifactError("pickle_grammar", "truncated pickle")
        b = pkl[p:p + n]
        p += n
        return b

    def integer(op):
        if op == b"K":
            return take(1)[0]
        if op == b"M":
            return struct.unpack("<H", take(2))[0]
        if op == b"J":
            return struct.unpack("<i", take(4))[0]
        return None

    if take(1) != b"\x80" or not 2 <= take(1)[0] <= 5:
        raise ArtifactError("pickle_grammar", "expected PROTO 2..5")
    if take(1) != b"]":
        raise ArtifactError("pickle_grammar", "expected EMPTY_LIST")
    ids = []
    op = take(1)
    if op == b"q":
        take(1)
        op = take(1)
    while op != b".":
        if op == b"(":
            op = take(1)
            while (v := integer(op)) is not None:
                ids.append(v)
                op = take(1)
            if op != b"e":
                raise ArtifactError("pickle_grammar", f"unexpected opcode {op!r} inside MARK group")
        else:
            v = integer(op)
            if v is None or take(1) != b"a":
                raise ArtifactError("pickle_grammar", f"unexpected opcode {op!r}")
            ids.append(v)
        op = take(1)
    if p != len(pkl):
        raise ArtifactError("pickle_grammar", f"{len(pkl) - p} bytes after STOP")
    return ids


def check_ids(ids: list[int], vocab_size: int) -> None:
    if not ids:
        raise ArtifactError("token_empty", "empty token map")
    bad = next((v for v in ids if not 0 <= v < vocab_size), None)
    if bad is not None:
        raise ArtifactError("token_range", f"id {bad} outside [0, {vocab_size})")
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ArtifactError("token_order", "ids are not strictly increasing")


def convert(data: bytes, vocab_size: int, *, limits: dict | None = None) -> bytes:
    ids = parse_int_list(find_data_pkl(data, limits_with(limits)))
    check_ids(ids, vocab_size)
    return struct.pack(f"<{len(ids)}I", *ids)


def vocab_size_of(config: bytes) -> int:
    v = parse_json(config)["text_config"]["vocab_size"]
    if type(v) is not int or v < 1:
        raise ValueError("config text_config.vocab_size must be a positive integer")
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("token_map", type=Path)
    p.add_argument("--config", type=Path, required=True, help="config.json carrying text_config.vocab_size")
    p.add_argument("--out", type=Path)
    p.add_argument("--limits", type=json.loads, default=None)
    args = p.parse_args(argv)
    try:
        out = convert(args.token_map.read_bytes(), vocab_size_of(args.config.read_bytes()), limits=args.limits)
    except (ValueError, OSError, KeyError, TypeError) as e:
        cls = getattr(e, "cls", "error")
        msg = str(e)
        sys.stderr.write(msg + "\n" if msg.startswith(cls + ": ") else f"{cls}: {msg}\n")
        return 2
    if args.out:
        args.out.write_bytes(out)
    else:
        sys.stdout.buffer.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
