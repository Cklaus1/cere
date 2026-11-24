"""Inventory HF safetensors checkpoints from their index and shard headers; check NVFP4-CSF scale streams.

    python3 scripts/native/ckpt_inventory.py groups ROOT
    python3 scripts/native/ckpt_inventory.py diff ROOT_A ROOT_B
    python3 scripts/native/ckpt_inventory.py csf-check ROOT [--every K]
    python3 scripts/native/ckpt_inventory.py conv-check SRC OUT CUR [--jobs N]

tasks/fp4local-spec.md (FL-01..FL-03, FL-11). Stdlib only, CPU only. `groups` and `diff` read config.json,
model.safetensors.index.json and each shard's header (8-byte little-endian length + JSON), never tensor data.
`groups` prints per tensor group (indices folded to N / E / S) its count, its "dtype [shape]" forms and bytes; `diff`
prints every config.json key that differs (dotted path, "<absent>" for a missing key) and the groups only in A, only
in B, or whose forms differ.

`csf-check` reads the routed-expert scale streams of an NVFP4-CSF checkpoint (modelopt `weight_scale_encoding: "csf"`;
codec "byte-window4-fixed-stream-u24-exceptions/1" of local-inference-lab/vllm
`vllm/model_executor/model_loader/nvfp4_csf_loader.py` @5009fa56 and b12x `b12x/_lib/quant/nvfp4_csf.py` @2cc7f66a).
The logical tensor `<name>` = E4M3 block scales [rows, cols] (cols = K/16) is stored as two tensors:
- `<name>.nvfp4_csf_fixed` U8 [rows/16, 16*(1 + cols/2)]: per 16-row slab, 16 row-base bytes (0..240), then the slab's
  16 rows x cols/2 bytes of 4-bit offsets, row-major, the even column in the low nibble; scale = base + offset.
- `<name>.nvfp4_csf_exceptions` U32 [n]: words (byte << 24) | position (row * cols + col), positions strictly
  increasing; the word's byte replaces the scale at that position, whose offset nibble is 0.
csf-check checks each pair is in the canonical form (bases <= 240; positions strictly increasing and in range; an
exception exactly where the scale falls outside [base, base+15], over a zero nibble), expands it, and counts NaN
(0x7F/0xFF), negative and zero scales. Prints one JSON object (totals, violations); exit 0 iff no violation, 1
otherwise, 2 on a bad checkpoint.
Prints aggregates only: no tensor bytes, no names beyond a violation's tensor name.

`conv-check` (FL-11) checks a `cerec convert` output OUT against its source SRC and the shipped checkpoint CUR. Each
output tensor is matched to the rule that made it: byte copy (sha256 equal), MXFP8 upcast (re-encoding the BF16 values
with the source UE8M0 scale gives the source E4M3 bytes), CSF expansion (equals `csf_expand` of the source streams),
F32 norm (equals the RNE to BF16; values the rounding changed are counted) or fused MTP experts (each expert's
[gate; up] / down slice equals the F32 NVFP4 dequant rounded to BF16; inexact values counted). The output's
name/dtype/shape set must equal CUR's text tensors, except the PLE table (`*.ple.ple_embedding.ngram_embedding.*`,
NVFP4 in OUT). Every non-vision source tensor must be used and every output tensor matched. Prints one JSON object of
counts; exit 0 iff `failures` is 0, 1 otherwise, 2 on a bad checkpoint.
An S3 output (`cerec convert --format s3`; OUT/CONVERSION.json `"format": "s3"`, absent = s2; tasks/fp4local-spec.md
S3) keeps the MXFP8 pairs and the NVFP4 MTP experts: `mxfp8_kept` (the F8_E4M3 weight and U8 scale byte-equal to the
source, canonical: scales in [10, 246], no E4M3 NaN byte) and `mtp_nvfp4` (each expert's slice of the fused packed
weight, block scales and globals byte-equal to the source pieces in [gate; up] order, no NaN block scale, finite
globals); the rest as S2. Its layout is CUR's with those tensors in their S3 form (`layout.s3_derived` counts them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import re
import struct
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
MAX_HEADER = 256 << 20
FIXED, EXCEPTIONS = ".nvfp4_csf_fixed", ".nvfp4_csf_exceptions"
MAX_BASE = 240
# Per row base: the low / high nibble of each packed byte plus the base (translate tables).
_LO = [bytes((b + (i & 15)) & 0xFF for i in range(256)) for b in range(256)]
_HI = [bytes((b + (i >> 4)) & 0xFF for i in range(256)) for b in range(256)]
_SIGN = bytes(0 if i < 0x80 else 1 for i in range(256))
_FOLD = ((re.compile(r"\.experts\.\d+\."), ".experts.E."), (re.compile(r"\.shard_\d+\."), ".shard_S."),
         (re.compile(r"(^|\.)(layers|blocks)\.\d+\."), r"\1\2.N."))


class InventoryError(Exception):
    """A checkpoint whose index or headers cannot be read."""


class CsfError(ValueError):
    """An NVFP4-CSF stream that is malformed or not in canonical form."""


def read_header(path: Path) -> dict:
    """The tensor entries of one safetensors file (`__metadata__` dropped)."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise InventoryError(f"{path}: shorter than a safetensors header")
        (n,) = struct.unpack("<Q", head)
        raw = f.read(n) if n <= MAX_HEADER else b""
    if n > MAX_HEADER or len(raw) != n:
        raise InventoryError(f"{path}: header length {n} exceeds the file or {MAX_HEADER}")
    try:
        header = json.loads(raw)
    except ValueError as exc:
        raise InventoryError(f"{path}: header is not JSON") from exc
    header.pop("__metadata__", None)
    return header


def load_inventory(root: Path) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """name -> (dtype, shape, shard) for every tensor of the index, from the shard headers."""
    root = Path(root)
    try:
        weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        raise InventoryError(f"{root}: no readable model.safetensors.index.json") from exc
    headers: dict[str, dict] = {}
    out = {}
    for name, shard in weight_map.items():
        if shard not in headers:
            path = root / shard
            if not path.is_file():
                raise InventoryError(f"{root}: shard {shard} of the index is missing")
            headers[shard] = read_header(path)
        entry = headers[shard].get(name)
        if entry is None:
            raise InventoryError(f"{root}: {name} is not in {shard}")
        out[name] = (entry["dtype"], tuple(entry["shape"]), shard)
    return out


def group_name(name: str) -> str:
    for pattern, repl in _FOLD:
        name = pattern.sub(repl, name)
    return name


def _nbytes(dtype: str, shape: Sequence[int]) -> int:
    n = DTYPE_BYTES[dtype]
    for s in shape:
        n *= s
    return n


def group_inventory(inv: dict[str, tuple[str, tuple[int, ...], str]]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for name, (dtype, shape, _) in sorted(inv.items()):
        g = groups.setdefault(group_name(name), {"count": 0, "forms": Counter(), "bytes": 0})
        g["count"] += 1
        g["forms"][f"{dtype} [{','.join(map(str, shape))}]"] += 1
        g["bytes"] += _nbytes(dtype, shape)
    return {k: {**v, "forms": dict(v["forms"])} for k, v in groups.items()}


def _flatten(value, prefix: str, out: dict) -> None:
    if isinstance(value, dict) and value:
        for k, v in value.items():
            _flatten(v, f"{prefix}.{k}" if prefix else k, out)
    else:
        out[prefix] = value


def config_diff(a: dict, b: dict) -> list[list]:
    fa, fb = {}, {}
    _flatten(a, "", fa)
    _flatten(b, "", fb)
    absent = "<absent>"
    return [[k, fa.get(k, absent), fb.get(k, absent)] for k in sorted(fa.keys() | fb.keys())
            if fa.get(k, absent) != fb.get(k, absent)]


def csf_geometry(fixed_shape: Sequence[int]) -> tuple[int, int]:
    """(rows, cols) of the logical scale plane of a `.nvfp4_csf_fixed` tensor of this shape."""
    if len(fixed_shape) != 2:
        raise CsfError(f"fixed stream must be rank 2, got {list(fixed_shape)}")
    slabs, width = fixed_shape
    if slabs <= 0 or width <= 16 or width % 16:
        raise CsfError(f"fixed stream shape {list(fixed_shape)} is not [rows/16, 16*(1+cols/2)]")
    return slabs * 16, (width // 16 - 1) * 2


def csf_expand(fixed: bytes, words: Sequence[int], rows: int, cols: int) -> bytes:
    """The plain E4M3 scale plane [rows, cols] (row-major bytes) of one CSF stream pair; no canonical-form checks."""
    width = 16 * (1 + cols // 2)
    if rows % 16 or cols % 2 or len(fixed) != rows // 16 * width:
        raise CsfError(f"fixed stream of {len(fixed)} bytes does not hold a [{rows}, {cols}] plane")
    half = cols // 2
    out = bytearray(rows * cols)
    for r in range(rows):
        slab, rr = divmod(r, 16)
        base = fixed[slab * width + rr]
        start = slab * width + 16 + rr * half
        packed = fixed[start:start + half]
        out[r * cols:(r + 1) * cols:2] = packed.translate(_LO[base])
        out[r * cols + 1:(r + 1) * cols:2] = packed.translate(_HI[base])
    n = rows * cols
    for w in words:
        p = w & 0xFFFFFF
        if p >= n:
            raise CsfError(f"exception position {p} outside the [{rows}, {cols}] plane")
        out[p] = w >> 24
    return bytes(out)


def csf_encode(plane: bytes, rows: int, cols: int, bases: bytes) -> tuple[bytes, list[int]]:
    """Canonical CSF streams of a plain plane for the given row bases (the inverse of csf_expand)."""
    if len(plane) != rows * cols or len(bases) != rows or rows % 16 or cols % 2:
        raise CsfError("plane, bases and geometry disagree")
    if rows * cols > 1 << 24:
        raise CsfError("plane exceeds the 24-bit exception positions")
    half = cols // 2
    fixed = bytearray()
    words = []
    for slab in range(rows // 16):
        fixed += bases[slab * 16:(slab + 1) * 16]
        for r in range(slab * 16, slab * 16 + 16):
            if bases[r] > MAX_BASE:
                raise CsfError(f"row base {bases[r]} above {MAX_BASE}")
            offsets = bytearray(cols)
            for c in range(cols):
                v = plane[r * cols + c]
                d = v - bases[r]
                if 0 <= d <= 15:
                    offsets[c] = d
                else:
                    words.append((v << 24) | (r * cols + c))
            fixed += bytes(offsets[2 * j] | (offsets[2 * j + 1] << 4) for j in range(half))
    return bytes(fixed), words


def csf_verify(fixed: bytes, words: Sequence[int], rows: int, cols: int) -> dict[str, int]:
    """Expand and check canonical form, and count special scales.

    Canonical (the streams csf_encode produces from the expanded plane and the stored bases): bases <= 240, positions
    strictly increasing and inside the plane, and at each exception a zero offset nibble and a byte outside
    [base, base+15]. Every other scale is base + nibble by construction, so these checks equal a re-encode."""
    width = 16 * (1 + cols // 2)
    if rows % 16 or cols % 2 or len(fixed) != rows // 16 * width:
        raise CsfError(f"fixed stream of {len(fixed)} bytes does not hold a [{rows}, {cols}] plane")
    top = max(fixed[s * width + i] for s in range(rows // 16) for i in range(16))
    if top > MAX_BASE:
        raise CsfError(f"row base {top} above {MAX_BASE}")
    half, prev = cols // 2, -1
    for w in words:
        p = w & 0xFFFFFF
        if p <= prev:
            raise CsfError("exception positions are not strictly increasing")
        if p >= rows * cols:
            raise CsfError(f"exception position {p} outside the [{rows}, {cols}] plane")
        prev = p
        r, c = divmod(p, cols)
        slab, rr = divmod(r, 16)
        base = fixed[slab * width + rr]
        code = fixed[slab * width + 16 + rr * half + c // 2]
        if (code >> 4 if c % 2 else code & 15) != 0:
            raise CsfError(f"exception at {p} over a non-zero offset nibble")
        if 0 <= (w >> 24) - base <= 15:
            raise CsfError(f"exception at {p} inside its row's window")
    plane = csf_expand(fixed, words, rows, cols)
    return {"scales": rows * cols, "exceptions": len(words),
            "nan": plane.count(0x7F) + plane.count(0xFF),
            "negative": len(plane) - plane.translate(_SIGN).count(0), "zero": plane.count(0)}


def csf_check(root: Path, every: int = 1) -> dict:
    root = Path(root)
    inv = load_inventory(root)
    names = sorted(n[:-len(FIXED)] for n in inv if n.endswith(FIXED))[::every]
    totals = Counter(tensors=0, scales=0, exceptions=0, nan=0, negative=0, zero=0, csf_bytes=0)
    violations = []
    maps: dict[str, tuple] = {}
    for name in names:
        ex = inv.get(name + EXCEPTIONS)
        dtype, shape, shard = inv[name + FIXED]
        if ex is None or dtype != "U8" or ex[0] != "U32" or ex[2] != shard or len(ex[1]) != 1:
            violations.append({"tensor": name, "error": "stream pair dtypes/shards are not U8 + U32 in one shard"})
            continue
        if shard not in maps:
            f = open(root / shard, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            (n,) = struct.unpack("<Q", mm[:8])
            maps[shard] = (f, mm, json.loads(mm[8:8 + n]), 8 + n)
        _, mm, header, base = maps[shard]

        def data(key: str) -> bytes:
            a, b = header[key]["data_offsets"]
            return mm[base + a:base + b]
        try:
            rows, cols = csf_geometry(shape)
            fixed = data(name + FIXED)
            raw = data(name + EXCEPTIONS)
            words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
            stats = csf_verify(fixed, words, rows, cols)
        except CsfError as exc:
            violations.append({"tensor": name, "error": str(exc)})
            continue
        totals.update(stats)
        totals["tensors"] += 1
        totals["csf_bytes"] += len(fixed) + len(raw)
    for f, mm, _, _ in maps.values():
        mm.close()
        f.close()
    return {**totals, "pairs_checked": len(names), "every": every, "violations": violations}


# ---- conv-check (FL-11): the `cerec convert` output against its source and the shipped checkpoint ----

PLE_TABLE = ".ple.ple_embedding.ngram_embedding."
_MTP_FUSED = re.compile(r"^(mtp\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")
MXFP8_MIN_SCALE, MXFP8_MAX_SCALE = 10, 246  # k2m::kMinScale / kMaxScale: every E4M3 x 2^(s-127) is a normal BF16
_CANON_SCALES = bytes(range(MXFP8_MIN_SCALE, MXFP8_MAX_SCALE + 1))


def has_e4m3_nan(b: bytes) -> bool:
    return b"\x7f" in b or b"\xff" in b


def mxfp8_scales_canonical(scales: bytes) -> bool:
    """S3: every UE8M0 byte in [MXFP8_MIN_SCALE, MXFP8_MAX_SCALE]."""
    return not scales.translate(None, _CANON_SCALES)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def e4m3_value(b: int) -> float | None:
    """An E4M3 (FN) byte's value; None for NaN (0x7F / 0xFF)."""
    e, m = (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return None
    mag = m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10)
    return -mag if b & 0x80 else mag


def f32_bits(x: float) -> int:
    """The bits of x rounded to F32 (round to nearest even); x must be finite in F32."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16_rne_bits(u: int) -> int:
    """F32 bits -> BF16 bits, round to nearest even (NaN not expected: the caller counts it as a mismatch)."""
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


_MX_ENC: dict[int, dict[int, int]] = {}


def mxfp8_encoder(scale: int) -> dict[int, int]:
    """BF16 bits -> the E4M3 byte b with e4m3(b) x 2^(scale-127) == that BF16 value exactly (MXFP8 re-encode)."""
    enc = _MX_ENC.get(scale)
    if enc is None:
        enc = {}
        if scale != 0xFF:
            for b in range(256):
                v = e4m3_value(b)
                if v is None:
                    continue
                v *= 2.0 ** (scale - 127)
                try:
                    u = f32_bits(v)
                except OverflowError:
                    continue
                if struct.unpack("<f", struct.pack("<I", u))[0] == v and u & 0xFFFF == 0 and u & 0x7F800000 != 0x7F800000:
                    enc[u >> 16] = b
        _MX_ENC[scale] = enc
    return enc


def mxfp8_reencode(bf16: bytes, scales: bytes, cols: int) -> bytes | None:
    """The E4M3 bytes a BF16 MXFP8 upcast came from (32-wide groups), or None when a value is not re-encodable."""
    words = memoryview(bf16).cast("H")
    out = bytearray(len(words))
    try:
        for g, s in enumerate(scales):
            out[g * 32:g * 32 + 32] = bytes(map(mxfp8_encoder(s).__getitem__, words[g * 32:g * 32 + 32]))
    except KeyError:
        return None
    return bytes(out)


def norm_rne(src: bytes) -> tuple[bytes, int]:
    """F32 norm weights -> BF16 bytes (RNE) and the number of values the rounding changed."""
    words = memoryview(src).cast("I")
    out = struct.pack(f"<{len(words)}H", *map(bf16_rne_bits, words))
    return out, sum(1 for u in words if u & 0xFFFF)


def nvfp4_dequant(packed: bytes, scales: bytes, global_scale: float) -> tuple[bytes, int]:
    """NVFP4 -> BF16 bytes: bf16_rne((e2m1 x e4m3 scale) x global) in F32, even column in the low nibble; and the number
    of values whose F32 product is not a BF16 value. Raises ValueError on a NaN block scale."""
    tables: dict[int, tuple[list[bytes], list[int]]] = {}
    out, inexact = [], 0
    for blk, sb in enumerate(scales):
        t = tables.get(sb)
        if t is None:
            s = e4m3_value(sb)
            if s is None:
                raise ValueError("NaN block scale")
            # e2m1 x scale is exact (6 significant bits) and x global is exact in a double: one rounding to F32.
            bits = [f32_bits(E2M1[c] * s * global_scale) for c in range(16)]
            vals = [struct.pack("<H", bf16_rne_bits(u)) for u in bits]
            inx = [int(u & 0xFFFF != 0) for u in bits]
            t = tables[sb] = ([vals[b & 15] + vals[b >> 4] for b in range(256)],
                              [inx[b & 15] + inx[b >> 4] for b in range(256)])
        block = packed[blk * 8:blk * 8 + 8]
        out.append(b"".join(map(t[0].__getitem__, block)))
        inexact += sum(map(t[1].__getitem__, block))
    return b"".join(out), inexact


class _Files:
    """Per-process mmaps of the safetensors files a check reads."""

    def __init__(self) -> None:
        self.maps: dict[Path, tuple] = {}

    def data(self, path: Path, name: str) -> bytes:
        if path not in self.maps:
            f = open(path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            (n,) = struct.unpack("<Q", mm[:8])
            self.maps[path] = (f, mm, json.loads(mm[8:8 + n]), 8 + n)
        _, mm, header, base = self.maps[path]
        a, b = header[name]["data_offsets"]
        return mm[base + a:base + b]


_FILES = _Files()


def _check_task(task: tuple) -> tuple[str, int, int, int]:
    """One unit of conv-check work -> (rule, mismatches, elements, inexact)."""
    rule, out_refs, src_refs, meta = task
    outs = [_FILES.data(*r) for r in out_refs]
    got = outs[0]
    src = [_FILES.data(*r) for r in src_refs]
    if rule == "copy":
        return rule, int(hashlib.sha256(got).digest() != hashlib.sha256(src[0]).digest()), len(got), 0
    if rule == "mxfp8_kept":
        bad = (outs[0] != src[0] or outs[1] != src[1] or has_e4m3_nan(outs[0])
               or not mxfp8_scales_canonical(outs[1]))
        return rule, int(bad), len(src[0]), 0
    if rule == "mtp_nvfp4":
        # One expert's slices of the fused weight / block scales / globals (meta = expert): the source pieces in
        # projection order ([gate, up] or [down]); no NaN block scale, finite globals.
        e, slices = meta, (src[0::3], src[1::3], src[2::3])
        bad = False
        for whole, parts in zip(outs, slices):
            w = b"".join(parts)
            bad |= whole[e * len(w):(e + 1) * len(w)] != w
        for scales, glob in zip(slices[1], slices[2]):
            bad |= has_e4m3_nan(scales) or not math.isfinite(struct.unpack("<f", glob)[0])
        return rule, int(bad), sum(len(p) * 2 for p in slices[0]), 0
    if rule == "mxfp8":
        return rule, int(mxfp8_reencode(got, src[1], meta) != src[0]), len(src[0]), 0
    if rule == "csf":
        rows, cols = meta
        try:
            plane = csf_expand(src[0], struct.unpack(f"<{len(src[1]) // 4}I", src[1]), rows, cols)
        except CsfError:
            return rule, 1, rows * cols, 0
        return rule, int(plane != got), rows * cols, 0
    if rule == "norm":
        want, inexact = norm_rne(src[0])
        return rule, int(want != got), len(src[0]) // 4, inexact
    # mtp: one expert's slice of a fused tensor; meta = (byte offset into the fused tensor, parts per projection).
    offset, nproj = meta
    want, inexact = [], 0
    for p in range(nproj):
        packed, scales, glob = src[3 * p:3 * p + 3]
        try:
            values, n = nvfp4_dequant(packed, scales, struct.unpack("<f", glob)[0])
        except (ValueError, OverflowError):
            return "mtp", 1, len(packed) * 2, 0
        want.append(values)
        inexact += n
    w = b"".join(want)
    return "mtp", int(got[offset:offset + len(w)] != w), len(w) // 2, inexact


def conversion_format(out: Path) -> str:
    """The `cerec convert` output format: OUT/CONVERSION.json `format`, absent (or no file) = "s2"."""
    path = Path(out) / "CONVERSION.json"
    fmt = json.loads(path.read_text()).get("format", "s2") if path.exists() else "s2"
    if fmt not in ("s2", "s3"):
        raise ValueError(f"{path}: unknown format {fmt!r}")
    return fmt


def s3_layout(c_inv: dict, s_inv: dict) -> tuple[dict, int]:
    """CUR's name -> (dtype, shape) with the S3-kept tensors in their S3 form, and how many CUR tensors changed."""
    want, derived = {}, 0
    for n, (dtype, shape, _) in c_inv.items():
        m = _MTP_FUSED.match(n)
        if s_inv.get(n, ("",))[0] == "F8_E4M3" and s_inv.get(n + "_scale", ("",))[0] == "U8" and len(shape) == 2:
            want[n] = ("F8_E4M3", tuple(shape))
            want[n + "_scale"] = ("U8", (shape[0], shape[1] // 32))
            derived += 1
        elif m and f"{m[1]}.0.gate_proj.weight" in s_inv and len(shape) == 3:
            e, rows, cols = shape
            want[n] = ("U8", (e, rows, cols // 2))
            want[n + "_scale"] = ("F8_E4M3", (e, rows, cols // 16))
            want[n + "_scale_2"] = ("F32", (e, 2) if m[2] == "gate_up_proj" else (e,))
            derived += 1
        else:
            want[n] = (dtype, tuple(shape))
    return want, derived


def conv_check(src: Path, out: Path, cur: Path, jobs: int = 1) -> dict:
    """FL-11: every output tensor of `cerec convert` against the rule that made it; the name/dtype/shape set against
    the shipped checkpoint (except the PLE table, which stays NVFP4; S3: CUR's layout with the kept tensors in S3
    form). Aggregates only."""
    src, out, cur = Path(src), Path(out), Path(cur)
    fmt = conversion_format(out)
    s_inv, o_inv, c_inv = load_inventory(src), load_inventory(out), load_inventory(cur)
    layout = {"only_out": 0, "only_cur": 0, "form_differs": 0, "ple_tensors_exempt": 0, "vision_in_out": 0}
    c_text = {n: (dt, tuple(sh)) for n, (dt, sh, _) in c_inv.items() if not n.startswith("model.visual.")}
    if fmt == "s3":
        c_text, layout["s3_derived"] = s3_layout({n: c_inv[n] for n in c_text}, s_inv)
    for n in o_inv.keys() | c_text.keys():
        if PLE_TABLE in n:
            layout["ple_tensors_exempt"] += 1
        elif n not in c_text:
            layout["vision_in_out" if n.startswith("model.visual.") else "only_out"] += 1
        elif n not in o_inv:
            layout["only_cur"] += 1
        elif (o_inv[n][0], tuple(o_inv[n][1])) != c_text[n]:
            layout["form_differs"] += 1

    def ref(root: Path, inv: dict, name: str) -> tuple[Path, str]:
        return root / inv[name][2], name

    tasks, used, unmapped, covered = [], set(), 0, set()
    for name, (dtype, shape, _) in sorted(o_inv.items()):
        if name in covered:
            continue
        m = _MTP_FUSED.match(name)
        if m and f"{m[1]}.0.gate_proj.weight" in s_inv:
            projs = ("gate", "up") if m[2] == "gate_up_proj" else ("down",)
            s3_names = [name, name + "_scale", name + "_scale_2"]
            if fmt == "s3" and not all(n in o_inv for n in s3_names):
                unmapped += 1
                continue
            if fmt == "s3":
                covered.update(s3_names[1:])
            per_expert = _nbytes(dtype, shape) // shape[0]
            for e in range(shape[0]):
                parts = [f"{m[1]}.{e}.{p}_proj.{k}" for p in projs for k in ("weight", "weight_scale", "weight_scale_2")]
                if any(p not in s_inv for p in parts):
                    unmapped += 1
                    break
                used.update(parts)
                srcs = [ref(src, s_inv, p) for p in parts]
                if fmt == "s3":
                    tasks.append(("mtp_nvfp4", [ref(out, o_inv, n) for n in s3_names], srcs, e))
                else:
                    tasks.append(("mtp", [ref(out, o_inv, name)], srcs, (e * per_expert, len(projs))))
            continue
        s = s_inv.get(name)
        sc = name + "_scale"
        if (fmt == "s3" and dtype == "F8_E4M3" and s is not None and s[:2] == (dtype, shape)
                and s_inv.get(sc, ("",))[0] == "U8" and o_inv.get(sc, ("",))[:2] == s_inv[sc][:2]):
            tasks.append(("mxfp8_kept", [ref(out, o_inv, name), ref(out, o_inv, sc)],
                          [ref(src, s_inv, name), ref(src, s_inv, sc)], None))
            used.update((name, sc))
            covered.add(sc)
        elif s is not None and s[:2] == (dtype, shape):
            tasks.append(("copy", [ref(out, o_inv, name)], [ref(src, s_inv, name)], None))
            used.add(name)
        elif name + FIXED in s_inv and name + EXCEPTIONS in s_inv and dtype == "F8_E4M3":
            tasks.append(("csf", [ref(out, o_inv, name)],
                          [ref(src, s_inv, name + FIXED), ref(src, s_inv, name + EXCEPTIONS)], tuple(shape)))
            used.update((name + FIXED, name + EXCEPTIONS))
        elif (fmt == "s2" and s is not None and s[0] == "F8_E4M3" and dtype == "BF16" and s[1] == shape
              and s_inv.get(name + "_scale", ("",))[0] == "U8"):
            tasks.append(("mxfp8", [ref(out, o_inv, name)], [ref(src, s_inv, name), ref(src, s_inv, name + "_scale")],
                          shape[1]))
            used.update((name, name + "_scale"))
        elif s is not None and s[0] == "F32" and dtype == "BF16" and s[1] == shape:
            tasks.append(("norm", [ref(out, o_inv, name)], [ref(src, s_inv, name)], None))
            used.add(name)
        else:
            unmapped += 1
    unused = sum(1 for n in s_inv if n not in used and not n.startswith("model.visual."))
    rules = ("copy", "mxfp8", "csf", "norm", "mtp") if fmt == "s2" else ("copy", "mxfp8_kept", "csf", "norm",
                                                                         "mtp_nvfp4")
    totals = {r: {"tensors": 0, "mismatched": 0, "elements": 0, "inexact": 0} for r in rules}
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            results = list(pool.imap_unordered(_check_task, tasks, chunksize=4))
    else:
        results = [_check_task(t) for t in tasks]
    for rule, bad, elements, inexact in results:
        t = totals[rule]
        t["tensors"] += 1
        t["mismatched"] += bad
        t["elements"] += elements
        t["inexact"] += inexact
    mtp = "mtp" if fmt == "s2" else "mtp_nvfp4"
    totals[mtp]["expert_slices"] = totals[mtp].pop("tensors")
    if fmt == "s3":
        del totals["mtp_nvfp4"]["inexact"], totals["mxfp8_kept"]["inexact"]
    failures = (sum(t["mismatched"] for t in totals.values()) + unmapped + unused
                + sum(v for k, v in layout.items() if k not in ("ple_tensors_exempt", "s3_derived")))
    return {"format": fmt, "layout": layout, "rules": totals, "unmapped_outputs": unmapped, "unused_sources": unused,
            "output_tensors": len(o_inv), "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups").add_argument("root", type=Path)
    d = sub.add_parser("diff")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    c = sub.add_parser("csf-check")
    c.add_argument("root", type=Path)
    c.add_argument("--every", type=int, default=1, help="check every K-th stream pair (default 1: all)")
    v = sub.add_parser("conv-check")
    v.add_argument("src", type=Path)
    v.add_argument("out", type=Path)
    v.add_argument("cur", type=Path)
    v.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "groups":
            print(json.dumps(group_inventory(load_inventory(args.root)), indent=1, sort_keys=True))
            return 0
        if args.cmd == "diff":
            ga, gb = (group_inventory(load_inventory(r)) for r in (args.a, args.b))
            cfg = [json.loads((r / "config.json").read_text()) for r in (args.a, args.b)]
            print(json.dumps({
                "config_diff": config_diff(*cfg),
                "groups_only_a": sorted(ga.keys() - gb.keys()),
                "groups_only_b": sorted(gb.keys() - ga.keys()),
                "groups_changed": {k: {"a": ga[k]["forms"], "b": gb[k]["forms"]}
                                   for k in sorted(ga.keys() & gb.keys()) if ga[k]["forms"] != gb[k]["forms"]},
            }, indent=1, sort_keys=True, default=str))
            return 0
        if args.cmd == "conv-check":
            if args.jobs < 1:
                parser.error("--jobs must be >= 1")
            report = conv_check(args.src, args.out, args.cur, args.jobs)
            print(json.dumps(report, indent=1, sort_keys=True))
            return 1 if report["failures"] else 0
        if args.every < 1:
            parser.error("--every must be >= 1")
        report = csf_check(args.root, args.every)
        print(json.dumps(report, indent=1, sort_keys=True))
        return 1 if report["violations"] else 0
    except (InventoryError, OSError, ValueError) as exc:
        print(f"ckpt_inventory: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
