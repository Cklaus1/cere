"""Inkling 2 Mini slice S7 (tasks/inkling-spec.md 3.6 and 7): the vision tower of family inkling2_loop
(executor/cuda/src/model/inkling/vision.h over kernels/inkling/vision.h and K2 GEMMs) vs the recorded reference tower
(transformers 5.24.0 `Inkling2VisionModel`, scripts/native/inkling_reference/run_vision.py, GPU0;
tests/fixtures/inkling2_reference/vision/).
Suite `inkling_kernels` (GPU0 only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels` on
the inkling-2-mini library with the checkpoint read-only at /model (the 422 `model.visual.*` tensors).

Every case reports row "INKLING_VISION". The tower is not bit-checked op by op (its gate is statistical, spec 3.6):
- processor (a precondition of every case, checked in setUpClass): the tokenizer sidecar's InklingProcessor
  (scripts/native/tokenizer_sidecar.py, feature `image`) on each recorded PNG (decoded here with zlib: 8-bit, no
  interlace, gray / RGB / RGBA without tRNS, as PIL's `L`/`RGB`/`RGBA` decode them) gives the reference processor's
  `pixel_values` bit for bit (sha256 of the FP32 [N, 40, 40, 3] bytes) and grid, and t * h * w / 4 placeholders = the
  reference processor's count (p08: 48); a mismatch fails the class. The processor's own PIL path is checked by
  scripts/native/inkling_reference/check_processor.py in the reference image.
- features: per image, CERE's merged features [tokens, 2,304] vs the reference tower in FP32 (stored as FP16; the
  storage costs ~2e-4 rel-L2): cosine >= 0.999 and rel-L2 <= 2 x the reference's own BF16-vs-FP32 rel-L2 on that image
  (BAR_SPREAD; recorded spread 0.015-0.038). Each image alone and all 4 in one call (cu_seqlens); a repeated call is
  bit-identical.
- fail-first: the tower without the class tokens, and with the RoPE's h and w positions swapped, each fail that bar on
  at least one recorded image (expected: both).
- binding: the ktest's tensor table (names and shapes) is exactly the checkpoint's `model.visual.*` set.

Entry points (libcere_ktest.so, executor/cuda/src/ktest/inkling/vision_ktest.cpp):
    int cere_ktest_inkling_vision_encode(const void *const *tensors, const void *pixels, const int32_t *grids,
                                         int64_t count, int64_t mutant, void *out, cudaStream_t stream);
    int cere_ktest_inkling_vision_tensor(int64_t i, char *name, int64_t cap, int64_t *dims);
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import struct
import unittest
import zlib

import torch
from safetensors import safe_open

import oracle

ROW = "INKLING_VISION"
FIXTURE = oracle.SRC / "tests/fixtures/inkling2_reference/vision"
SIDECAR = oracle.SRC / "scripts/native/tokenizer_sidecar.py"
TENSORS = 422
OUT_HIDDEN = 2304
MIN_COSINE = 0.999
BAR_SPREAD = 2.0
MUTANTS = {"no_class": 1, "rope_axes_swapped": 2}


def _sidecar():
    spec = importlib.util.spec_from_file_location("inkling_tokenizer_sidecar", SIDECAR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


class _Png:
    """The PIL-image surface InklingProcessor.rgb reads (`has_transparency_data`, `convert`, `size`, `tobytes`) of an
    8-bit, non-interlaced gray (L), RGB or RGBA PNG without tRNS/PLTE, decoded with zlib (PNG spec section 9 filters).
    `convert` is the identity or L -> RGB (each gray byte repeated 3 times), as PIL does for these modes."""

    MODES = {0: "L", 2: "RGB", 6: "RGBA"}

    def __init__(self, data: bytes, mode: str | None = None, raw: bytes | None = None):
        if raw is not None:
            self.size, self.mode, self.raw = data, mode, raw
            return
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise oracle.OracleError("not a PNG")
        pos, idat, header = 8, [], None
        while pos < len(data):
            length, kind = struct.unpack(">I4s", data[pos:pos + 8])
            body = data[pos + 8:pos + 8 + length]
            pos += 12 + length
            if kind == b"IHDR":
                header = struct.unpack(">IIBBBBB", body)
            elif kind == b"IDAT":
                idat.append(body)
            elif kind in (b"tRNS", b"PLTE"):
                raise oracle.OracleError(f"PNG chunk {kind.decode()} is not decoded here")
        width, height, depth, color, _, _, interlace = header
        if depth != 8 or interlace != 0 or color not in self.MODES:
            raise oracle.OracleError(f"PNG depth {depth} color {color} interlace {interlace} is not decoded here")
        self.size, self.mode = (width, height), self.MODES[color]
        bpp = len(self.mode)
        stride = width * bpp
        packed = zlib.decompress(b"".join(idat))
        out, prev = bytearray(), bytearray(stride)
        for y in range(height):
            kind, row = packed[y * (stride + 1)], bytearray(packed[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                b = prev[i]
                row[i] = (row[i] + (0, a, b, (a + b) // 2, _paeth(a, b, c))[kind]) & 0xFF
            out += row
            prev = row
        self.raw = bytes(out)

    @property
    def has_transparency_data(self) -> bool:
        return self.mode == "RGBA"

    def convert(self, mode: str) -> "_Png":
        if mode == self.mode:
            return self
        if (self.mode, mode) == ("L", "RGB"):
            return _Png(self.size, mode, bytes(v for g in self.raw for v in (g, g, g)))
        raise oracle.OracleError(f"PNG convert {self.mode} -> {mode} is not decoded here")

    def tobytes(self) -> bytes:
        return self.raw


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


class InklingVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((FIXTURE / "vision.json").read_text())
        cls.order = cls.record["order"]
        with safe_open(str(FIXTURE / "vision.safetensors"), framework="pt", device="cpu") as f:
            cls.reference = {n: f.get_tensor(f"f32.{n}").float() for n in cls.order}
        sidecar = _sidecar()
        config = json.loads((oracle.MODEL_DIR / "processor_config.json").read_text())["image_processor"]
        cls.processor = sidecar.InklingProcessor(torch, config)
        cls.pixels, cls.grids = {}, {}
        for n in cls.order:
            rec = cls.record["images"][n]
            cells, rows, cols = cls.processor.cells(_Png((FIXTURE / f"{n}.png").read_bytes()))
            digest = hashlib.sha256(cells.contiguous().numpy().tobytes()).hexdigest()
            same = (digest == rec["pixel_values_sha256"] and [1, rows, cols] == rec["grid_thw"]
                    and rows * cols == rec["placeholders"])
            oracle.report_row(ROW, f"processor {n}", phase="result", tokens=rows * cols, passed=same)
            if not same:
                raise oracle.OracleError(f"{n}: processor output differs from the reference's")
            cls.pixels[n] = cls.processor.patch_rows(cells).to(torch.bfloat16).to(oracle.DEVICE).contiguous()
            cls.grids[n] = (2 * rows, 2 * cols)
        if cls.record["images"]["p08"]["placeholders"] != 48:
            raise oracle.OracleError("p08's recorded placeholder count is not 48")
        cls.names, cls.shapes = cls.tensor_table()
        cls.weights = [oracle.real_tensor(n).contiguous() for n in cls.names]
        cls.table = (ctypes.c_void_p * TENSORS)(*[t.data_ptr() for t in cls.weights])

    @classmethod
    def tensor_table(cls) -> tuple[list[str], list[tuple[int, ...]]]:
        """The ktest's tensor table: (names, shapes) from cere_ktest_inkling_vision_tensor."""
        names, shapes = [], []
        name_buf = ctypes.create_string_buffer(256)
        dims = (ctypes.c_int64 * 2)()
        fn = oracle.ktest("cere_ktest_inkling_vision_tensor")
        for i in range(TENSORS):
            ndim = fn(ctypes.c_int64(i), name_buf, ctypes.c_int64(len(name_buf)), dims)
            if ndim <= 0:
                raise oracle.OracleError(f"cere_ktest_inkling_vision_tensor({i}) returned {ndim}")
            names.append(name_buf.value.decode())
            shapes.append(tuple(dims[d] for d in range(ndim)))
        return names, shapes

    def encode(self, names: list[str], mutant: int = 0) -> list[torch.Tensor]:
        grids = [self.grids[n] for n in names]
        pixels = torch.cat([self.pixels[n] for n in names]).contiguous()
        tokens = [h * w // 4 for h, w in grids]
        out = oracle.empty((sum(tokens), OUT_HIDDEN), torch.bfloat16)
        flat = (ctypes.c_int32 * (2 * len(grids)))(*[v for g in grids for v in g])
        stream = torch.cuda.current_stream(oracle.DEVICE)
        rc = oracle.ktest("cere_ktest_inkling_vision_encode")(
            self.table, ctypes.c_void_p(pixels.data_ptr()), flat, ctypes.c_int64(len(grids)),
            ctypes.c_int64(mutant), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(stream.cuda_stream))
        stream.synchronize()
        if rc != 0:
            raise oracle.OracleError(f"cere_ktest_inkling_vision_encode returned {rc}")
        return list(out.float().cpu().split(tokens))

    def measure(self, name: str, ours: torch.Tensor, case: str) -> tuple[bool, float, float, float]:
        ref = self.reference[name]
        cosine, rel = _cosine(ours, ref), oracle.rel_l2(ours.double(), ref.double())
        limit = BAR_SPREAD * self.record["images"][name]["bf16_vs_f32_cos_rel"][1]
        passed = cosine >= MIN_COSINE and rel <= limit
        oracle.report_row(ROW, f"{case} {name}", phase="measure", cosine=cosine, rel_l2=rel, limit=limit,
                          min_cosine=MIN_COSINE, passed=passed)
        return passed, cosine, rel, limit

    def test_binding(self):
        with open(oracle.MODEL_DIR / "model.safetensors.index.json") as f:
            visual = {n for n in json.load(f)["weight_map"] if n.startswith("model.visual.")}
        names, shapes = self.tensor_table()  # queried again: the table is a pure function of the library
        self.assertEqual((names, shapes), (self.names, self.shapes))
        self.assertEqual(len(set(names)), TENSORS)
        self.assertEqual(set(names), visual)
        for name, shape, t in zip(names, shapes, self.weights):
            self.assertEqual(tuple(t.shape), shape, name)
            self.assertEqual(t.dtype, torch.bfloat16, name)
        oracle.report_row(ROW, "binding", phase="count", count=len(visual), expected=TENSORS, passed=True)

    def test_features(self):
        for n in self.order:
            passed, cosine, rel, limit = self.measure(n, self.encode([n])[0], "single")
            self.assertTrue(passed, f"{n}: cosine {cosine:.6f} rel-L2 {rel:.5f} (limit {limit:.5f})")

    def test_batched_and_repeat(self):
        first = self.encode(self.order)
        for n, ours in zip(self.order, first):
            passed, cosine, rel, limit = self.measure(n, ours, "batched")
            self.assertTrue(passed, f"{n} batched: cosine {cosine:.6f} rel-L2 {rel:.5f} (limit {limit:.5f})")
        again = self.encode(self.order)
        for n, a, b in zip(self.order, first, again):
            oracle.check_bit_exact(self, ROW, f"repeat {n}", a, b)

    def test_fail_first(self):
        for case, mutant in MUTANTS.items():
            failed = sum(not self.measure(n, ours, f"mutant {case}")[0]
                         for n, ours in zip(self.order, self.encode(self.order, mutant)))
            oracle.check_count(self, ROW, f"mutant {case} refused", int(failed > 0), expected=1)


if __name__ == "__main__":
    unittest.main()
