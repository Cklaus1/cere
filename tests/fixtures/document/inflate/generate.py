#!/usr/bin/env python3
"""Writes the zlib fixtures of crates/cere-runtime/src/document/inflate.rs's round-trip test (DocInput DI-04c).

The source is SOURCE_BYTES bytes from a seeded xorshift32 (words, random bytes, runs and back references, so the
streams use stored, fixed and dynamic blocks, long matches and far distances). The Rust test rebuilds the same
source with the same generator. Each fixture is that source compressed by Python's zlib once and checked in:
`level0.zz`, `level1.zz`, `level6.zz`, `level9.zz` and `fixed.zz` (level 6, strategy Z_FIXED). Run from the
repository root only to regenerate them (another zlib may emit other, equally valid, streams)."""
from __future__ import annotations

import zlib
from pathlib import Path

SOURCE_BYTES = 66_000
SEED = 0x2545F491
WORDS = [b"page", b"text", b"layer", b"document", b"stream", b"object", b"the", b"of", b"and", b"PDF", b"CERE",
         b"inflate", b"Huffman", b"code", b"length", b"distance"]


def source() -> bytes:
    x = SEED

    def rand() -> int:
        nonlocal x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        return x

    out = bytearray()
    while len(out) < SOURCE_BYTES:
        kind = rand() % 10
        if kind < 7:
            out += WORDS[rand() % len(WORDS)]
            out += b"\n" if rand() % 8 == 0 else b" "
        elif kind == 7:
            out += bytes(rand() & 0xFF for _ in range(64))
        elif kind == 8:
            out += bytes([rand() & 0xFF]) * (rand() % 300)
        elif out:
            distance = 1 + rand() % min(len(out), 32768)
            for _ in range(3 + rand() % 256):
                out.append(out[-distance])
    return bytes(out[:SOURCE_BYTES])


def main() -> None:
    here = Path(__file__).resolve().parent
    data = source()
    for level in (0, 1, 6, 9):
        (here / f"level{level}.zz").write_bytes(zlib.compress(data, level))
    fixed = zlib.compressobj(6, zlib.DEFLATED, 15, 8, zlib.Z_FIXED)
    (here / "fixed.zz").write_bytes(fixed.compress(data) + fixed.flush())


if __name__ == "__main__":
    main()
