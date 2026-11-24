"""tasks/fp4local-spec.md FL-01..FL-03: scripts/native/ckpt_inventory.py on synthetic checkpoints (headers, config
diff, tensor groups, NVFP4-CSF scale decode and canonical re-encode). No model bytes: every tensor is generated here."""
from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/native"))
import ckpt_inventory as ci  # noqa: E402


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def fixture_plane(rows: int, columns: int, expert: int) -> tuple[bytes, bytes]:
    """The b12x codec-0 test fixture (tests/quantization/test_nvfp4_csf.py `_fixture`): a per-row palette
    base + (column % 16), every 97th scale replaced by (index % 256). Returns (plane, row bases)."""
    bases = bytes((r + expert * 31) % 231 for r in range(rows))
    plane = bytearray(bases[r] + c % 16 for r in range(rows) for c in range(columns))
    for i in range(0, rows * columns, 97):
        plane[i] = i % 256
    return bytes(plane), bases


def u32(words: list[int]) -> bytes:
    return struct.pack(f"<{len(words)}I", *words)


class Header(unittest.TestCase):
    def test_reads_header_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.safetensors"
            write_safetensors(p, {"x.weight": ("BF16", [2, 3], bytes(12))})
            h = ci.read_header(p)
            self.assertEqual(h["x.weight"]["shape"], [2, 3])
            self.assertNotIn("__metadata__", h)
            p.write_bytes(struct.pack("<Q", 1 << 20) + b"{}")
            with self.assertRaises(ci.InventoryError):
                ci.read_header(p)


class Groups(unittest.TestCase):
    def test_group_names_normalise_indices(self):
        self.assertEqual(ci.group_name("model.language_model.layers.17.mlp.experts.311.up_proj.weight"),
                         "model.language_model.layers.N.mlp.experts.E.up_proj.weight")
        self.assertEqual(ci.group_name("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_63.weight"),
                         "model.language_model.layers.N.ple.ple_embedding.ngram_embedding.shard_S.weight")
        self.assertEqual(ci.group_name("model.visual.blocks.26.attn.qkv.weight"), "model.visual.blocks.N.attn.qkv.weight")
        self.assertEqual(ci.group_name("lm_head.weight"), "lm_head.weight")

    def test_inventory_and_diff_from_index_and_headers(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            for root, dtype, extra in ((a, "BF16", {}), (b, "F8_E4M3", {"layers.0.q_proj.weight_scale": ("U8", [4, 1], bytes(4))})):
                (root / "tensors").mkdir(parents=True)
                t0 = {f"layers.{i}.q_proj.weight": (dtype, [4, 32], bytes(128 * (2 if dtype == "BF16" else 1)))
                      for i in range(2)}
                write_safetensors(root / "tensors/m-1.safetensors", {**t0, **extra})
                write_safetensors(root / "tensors/m-2.safetensors", {"lm_head.weight": ("BF16", [8, 4], bytes(64))})
                wm = {n: "tensors/m-1.safetensors" for n in {**t0, **extra}} | {"lm_head.weight": "tensors/m-2.safetensors"}
                (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
                cfg = {"text_config": {"num_hidden_layers": 2, "ple_embedding_dtype": "float8_e4m3fn" if root == a else "nvfp4"}}
                if root == b:
                    cfg["quantization_config"] = {"quant_algo": "MIXED_PRECISION"}
                (root / "config.json").write_text(json.dumps(cfg))
            inv = ci.load_inventory(a)
            self.assertEqual(inv["lm_head.weight"], ("BF16", (8, 4), "tensors/m-2.safetensors"))
            groups = ci.group_inventory(inv)
            self.assertEqual(groups["layers.N.q_proj.weight"], {"count": 2, "forms": {"BF16 [4,32]": 2}, "bytes": 512})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ci.main(["diff", str(a), str(b)]), 0)
            rep = json.loads(out.getvalue())
            self.assertEqual(rep["config_diff"], [
                ["quantization_config.quant_algo", "<absent>", "MIXED_PRECISION"],
                ["text_config.ple_embedding_dtype", "float8_e4m3fn", "nvfp4"],
            ])
            self.assertEqual(rep["groups_only_b"], ["layers.N.q_proj.weight_scale"])
            self.assertEqual(rep["groups_changed"]["layers.N.q_proj.weight"],
                             {"a": {"BF16 [4,32]": 2}, "b": {"F8_E4M3 [4,32]": 2}})
            self.assertNotIn("lm_head.weight", rep["groups_changed"])

    def test_missing_shard_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "tensors/none.safetensors"}}))
            with self.assertRaises(ci.InventoryError):
                ci.load_inventory(root)


class Csf(unittest.TestCase):
    def test_fixture_round_trip_is_exact(self):
        for rows, columns, expert in ((16, 32, 0), (64, 160, 1), (128, 40, 3)):
            plane, bases = fixture_plane(rows, columns, expert)
            fixed, words = ci.csf_encode(plane, rows, columns, bases)
            self.assertEqual(len(fixed), rows // 16 * 16 * (1 + columns // 2))
            self.assertTrue(words, "the fixture has exceptions")
            self.assertEqual(ci.csf_geometry((rows // 16, 16 * (1 + columns // 2))), (rows, columns))
            self.assertEqual(ci.csf_expand(fixed, words, rows, columns), plane)
            stats = ci.csf_verify(fixed, words, rows, columns)
            self.assertEqual(stats, {"scales": rows * columns, "exceptions": len(words),
                                     "nan": plane.count(0x7F) + plane.count(0xFF),
                                     "negative": sum(1 for x in plane if x & 0x80), "zero": plane.count(0)})

    def test_low_nibble_is_the_even_column(self):
        rows, columns = 16, 2
        fixed = bytearray(16 * 2)
        fixed[0] = 0x30          # row 0 base
        fixed[16] = 0x21         # row 0: column 0 -> base + 1, column 1 -> base + 2
        for r in range(1, 16):
            fixed[r] = 0x38
        self.assertEqual(ci.csf_expand(bytes(fixed), [], rows, columns)[:2], bytes([0x31, 0x32]))

    def test_exception_word_carries_the_whole_byte(self):
        plane, bases = fixture_plane(16, 32, 0)
        fixed, words = ci.csf_encode(plane, 16, 32, bases)
        for w in words:
            self.assertEqual(plane[w & 0xFFFFFF], w >> 24)

    def test_rejects_non_canonical_or_corrupt_streams(self):
        rows, columns = 32, 32
        plane, bases = fixture_plane(rows, columns, 2)
        fixed, words = ci.csf_encode(plane, rows, columns, bases)
        bad_base = bytearray(fixed)
        bad_base[0] = 241
        cases = {
            "base": (bytes(bad_base), words),
            "unsorted": (fixed, [words[1], words[0]] + words[2:]),
            "range": (fixed, words[:-1] + [(words[-1] & 0xFF000000) | rows * columns]),
            "inside window": (fixed, sorted(words + [((bases[31] + 1) << 24) | (31 * columns + 16)],
                                            key=lambda w: w & 0xFFFFFF)),
        }
        nib = bytearray(fixed)
        p = words[0] & 0xFFFFFF
        r, c = divmod(p, columns)
        nib[(r // 16) * 16 * (1 + columns // 2) + 16 + (r % 16) * (columns // 2) + c // 2] |= 0x0F if c % 2 == 0 else 0xF0
        cases["nibble"] = (bytes(nib), words)
        for name, (f, w) in cases.items():
            with self.subTest(name), self.assertRaises(ci.CsfError):
                ci.csf_verify(f, w, rows, columns)
        with self.assertRaises(ci.CsfError):
            ci.csf_geometry((2, 16 * 21 + 1))

    def test_csf_check_command_on_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tensors").mkdir()
            tensors, wm = {}, {}
            for e in range(3):
                plane, bases = fixture_plane(32, 40, e)
                fixed, words = ci.csf_encode(plane, 32, 40, bases)
                n = f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight_scale"
                tensors[n + ".nvfp4_csf_fixed"] = ("U8", [2, 16 * 21], fixed)
                tensors[n + ".nvfp4_csf_exceptions"] = ("U32", [len(words)], u32(words))
            write_safetensors(root / "tensors/m-1.safetensors", tensors)
            wm = {n: "tensors/m-1.safetensors" for n in tensors}
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ci.main(["csf-check", str(root)]), 0)
            rep = json.loads(out.getvalue())
            self.assertEqual((rep["tensors"], rep["scales"], rep["violations"]), (3, 3 * 32 * 40, []))
            self.assertEqual(rep["csf_bytes"], sum(len(t[2]) for t in tensors.values()))
            # Corrupt one stream: a base above 240.
            name = next(iter(tensors))
            dtype, shape, fixed = tensors[name]
            tensors[name] = (dtype, shape, b"\xf5" + fixed[1:])
            write_safetensors(root / "tensors/m-1.safetensors", tensors)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ci.main(["csf-check", str(root)]), 1)
            self.assertEqual(len(json.loads(out.getvalue())["violations"]), 1)


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rne(u: int) -> int:
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


def write_checkpoint(root: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    (root / "tensors").mkdir(parents=True)
    write_safetensors(root / "tensors/m-1.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "tensors/m-1.safetensors" for n in tensors}}))


L = "model.language_model.layers.0"
PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def nvfp4_expected(packed: bytes, scales: bytes, g: float, cols: int) -> tuple[bytes, int]:
    """Element by element: bf16_rne(f32(e2m1 x scale x global)), even column in the low nibble."""
    out, inexact = [], 0
    for i in range(len(packed) * 2):
        code = packed[i // 2] >> 4 if i % 2 else packed[i // 2] & 15
        r, c = divmod(i, cols)
        u = f32(E2M1[code] * ci.e4m3_value(scales[r * (cols // 16) + c // 16]) * g)
        inexact += u & 0xFFFF != 0
        out.append(rne(u))
    return struct.pack(f"<{len(out)}H", *out), inexact


def conversion_triplet(root: Path) -> tuple[Path, Path, Path, dict, dict]:
    """A synthetic FP4-Local source, its CERE-layout conversion (built here, rule by rule) and a shipped-layout CUR."""
    src, out = {}, {}
    src["lm_head.weight"] = out["lm_head.weight"] = ("BF16", [2, 4], bytes(range(16)))
    # MXFP8 [2, 64], scales [2, 2]: normals, subnormals (0x01..0x07), -0 (0x80); no NaN (even bytes only).
    w = bytes((i * 37) & 0xFE for i in range(128))
    s = bytes([127, 100, 140, 120])
    src[f"{L}.self_attn.q_proj.weight"] = ("F8_E4M3", [2, 64], w)
    src[f"{L}.self_attn.q_proj.weight_scale"] = ("U8", [2, 2], s)
    bf = []
    for i, b in enumerate(w):
        u = f32(ci.e4m3_value(b) * 2.0 ** (s[i // 32] - 127))
        assert u & 0xFFFF == 0
        bf.append(u >> 16)
    out[f"{L}.self_attn.q_proj.weight"] = ("BF16", [2, 64], struct.pack("<128H", *bf))
    # CSF routed scale plane [32, 40].
    plane, bases = fixture_plane(32, 40, 1)
    fixed, words = ci.csf_encode(plane, 32, 40, bases)
    name = f"{L}.mlp.experts.0.down_proj.weight_scale"
    src[name + ci.FIXED] = ("U8", [2, 16 * 21], fixed)
    src[name + ci.EXCEPTIONS] = ("U32", [len(words)], u32(words))
    out[name] = ("F8_E4M3", [32, 40], plane)
    # F32 norm: 1.0 exact, 1 + 2^-8 a tie (-> 1.0), 0.3 inexact.
    norm = [1.0, 1.00390625, 0.3]
    src[f"{L}.attn_hyper_connection.hc_norm.weight"] = ("F32", [3], struct.pack("<3f", *norm))
    out[f"{L}.attn_hyper_connection.hc_norm.weight"] = ("BF16", [3], struct.pack("<3H", *(rne(f32(x)) for x in norm)))
    # MTP experts: I = 16, H = 32, 2 experts, global 0.1 (inexact products).
    gate_up, down, mtp_inexact = [], [], 0
    for e in range(2):
        for k, (proj, rows, cols) in enumerate((("gate", 16, 32), ("up", 16, 32), ("down", 32, 16))):
            packed = bytes((e * 7 + k * 3 + i) % 256 for i in range(rows * cols // 2))
            scales = bytes(0x30 + (i + e + k) % 9 for i in range(rows * cols // 16))
            base = f"mtp.layers.0.mlp.experts.{e}.{proj}_proj"
            src[base + ".weight"] = ("U8", [rows, cols // 2], packed)
            src[base + ".weight_scale"] = ("F8_E4M3", [rows, cols // 16], scales)
            src[base + ".weight_scale_2"] = ("F32", [], struct.pack("<f", 0.1))
            values, n = nvfp4_expected(packed, scales, struct.unpack("<f", struct.pack("<f", 0.1))[0], cols)
            mtp_inexact += n
            (down if proj == "down" else gate_up).append(values)
    out["mtp.layers.0.mlp.experts.gate_up_proj"] = ("BF16", [2, 32, 32], b"".join(gate_up))
    out["mtp.layers.0.mlp.experts.down_proj"] = ("BF16", [2, 32, 16], b"".join(down))
    # PLE table: NVFP4, copied; CUR has it as FP8 + BF16 scale.
    for n, t in {f"{PLE}.shard_0.weight": ("U8", [2, 80], bytes(160)),
                 f"{PLE}.shard_0.weight_scale": ("F8_E4M3", [2, 10], bytes(range(20))),
                 f"{PLE}.weight_scale_2": ("F32", [1], struct.pack("<f", 0.5))}.items():
        src[n] = out[n] = t
    src["model.visual.patch_embed.weight"] = ("BF16", [2], bytes(4))
    cur = {n: (dt, sh, bytes(len(b))) for n, (dt, sh, b) in out.items() if ".ngram_embedding." not in n}
    cur[f"{PLE}.shard_0.weight"] = ("F8_E4M3", [2, 160], bytes(320))
    cur[f"{PLE}.weight_scale"] = ("BF16", [1], bytes(2))
    cur["model.visual.patch_embed.weight"] = ("BF16", [2], bytes(4))
    paths = root / "src", root / "out", root / "cur"
    for p, t in zip(paths, (src, out, cur)):
        write_checkpoint(p, t)
    return (*paths, out, {"mtp_inexact": mtp_inexact})


class ConvCheck(unittest.TestCase):
    def run_check(self, src: Path, out: Path, cur: Path, jobs: int = 1) -> tuple[int, dict]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ci.main(["conv-check", str(src), str(out), str(cur), "--jobs", str(jobs)])
        return rc, json.loads(buf.getvalue())

    def test_mxfp8_reencode_round_trips_subnormals_and_negative_zero(self):
        w = bytes([0x00, 0x80, 0x01, 0x81, 0x07, 0x38, 0x7E, 0xFE] * 4)
        bf = struct.pack("<32H", *(f32(ci.e4m3_value(b) * 2.0 ** -20) >> 16 for b in w))
        self.assertEqual(ci.mxfp8_reencode(bf, bytes([107]), 32), w)
        self.assertIsNone(ci.mxfp8_reencode(struct.pack("<32H", *[0x3F81] * 32), bytes([127]), 32),
                          "1 + 2^-7 is not an E4M3 value at scale 2^0")
        self.assertEqual(ci.mxfp8_encoder(0xFF), {}, "scale 0xFF is NaN")

    def test_norm_rne_ties_to_even_and_counts_changes(self):
        got, inexact = ci.norm_rne(struct.pack("<4f", 1.0, 1.00390625, 1.01171875, 0.3))
        self.assertEqual(struct.unpack("<4H", got), (0x3F80, 0x3F80, 0x3F82, 0x3E9A))
        self.assertEqual(inexact, 3)

    def test_nvfp4_dequant_matches_elementwise_reference(self):
        packed, scales = bytes(range(0, 256, 4)), bytes([0x30, 0x31, 0x38, 0x00, 0x3F, 0xB8, 0x40, 0x29])
        g = struct.unpack("<f", struct.pack("<f", 0.1))[0]
        self.assertEqual(ci.nvfp4_dequant(packed, scales, g), nvfp4_expected(packed, scales, g, 32))
        with self.assertRaises(ValueError):
            ci.nvfp4_dequant(bytes(8), b"\x7f", 1.0)

    def test_a_correct_conversion_passes_with_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, cur, _, want = conversion_triplet(Path(d))
            for jobs in (1, 2):
                rc, rep = self.run_check(src, out, cur, jobs)
                self.assertEqual((rc, rep["failures"]), (0, 0), rep)
                r = rep["rules"]
                self.assertEqual({k: r[k].get("tensors", r[k].get("expert_slices")) for k in r},
                                 {"copy": 4, "mxfp8": 1, "csf": 1, "norm": 1, "mtp": 4})
                self.assertEqual(r["norm"]["inexact"], 2)
                self.assertEqual(r["mtp"]["inexact"], want["mtp_inexact"])
                self.assertGreater(want["mtp_inexact"], 0)
                self.assertEqual(rep["layout"]["ple_tensors_exempt"], 4, "3 NVFP4 names + CUR's BF16 weight_scale")
                self.assertEqual(rep["output_tensors"], 9)

    def test_each_rule_detects_a_one_byte_mismatch(self):
        cases = {"lm_head.weight": "copy", f"{L}.self_attn.q_proj.weight": "mxfp8",
                 f"{L}.mlp.experts.0.down_proj.weight_scale": "csf", f"{L}.attn_hyper_connection.hc_norm.weight": "norm",
                 "mtp.layers.0.mlp.experts.gate_up_proj": "mtp", f"{PLE}.shard_0.weight_scale": "copy"}
        for name, rule in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                src, out, cur, tensors, _ = conversion_triplet(Path(d))
                dt, sh, b = tensors[name]
                tensors[name] = (dt, sh, bytes([b[0] ^ 0x01]) + b[1:])
                write_safetensors(out / "tensors/m-1.safetensors", tensors)
                rc, rep = self.run_check(src, out, cur)
                self.assertEqual((rc, rep["failures"], rep["rules"][rule]["mismatched"]), (1, 1, 1))

    def test_layout_and_coverage_failures(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, cur, tensors, _ = conversion_triplet(Path(d))
            # A vision tensor in the output, a dtype that differs from CUR, a source tensor left unused.
            tensors["model.visual.patch_embed.weight"] = ("BF16", [2], bytes(4))
            tensors["lm_head.weight"] = ("F16", [2, 4], bytes(range(16)))
            del tensors[f"{L}.attn_hyper_connection.hc_norm.weight"]
            write_safetensors(out / "tensors/m-1.safetensors", tensors)
            (out / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {n: "tensors/m-1.safetensors" for n in tensors}}))
            rc, rep = self.run_check(src, out, cur)
            self.assertEqual(rc, 1)
            lay = rep["layout"]
            self.assertEqual((lay["vision_in_out"], lay["form_differs"], lay["only_cur"]), (1, 1, 1))
            # The F16 lm_head matches no rule, so its source and the dropped norm's source are used by none.
            self.assertEqual((rep["unmapped_outputs"], rep["unused_sources"]), (1, 2))


MTP = "mtp.layers.0.mlp.experts"
QW = f"{L}.self_attn.q_proj.weight"


def read_tensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    return {name: (e["dtype"], e["shape"], raw[8 + n + e["data_offsets"][0]:8 + n + e["data_offsets"][1]])
            for name, e in json.loads(raw[8:8 + n]).items() if name != "__metadata__"}


def conversion_triplet_s3(root: Path) -> tuple[Path, Path, Path, dict]:
    """The S3 (`cerec convert --format s3`) output of the same source: MXFP8 pair and NVFP4 MTP pieces kept."""
    src, out, cur, tensors, _ = conversion_triplet(root)
    src_t = read_tensors(src / "tensors/m-1.safetensors")
    tensors[QW] = src_t[QW]
    tensors[QW + "_scale"] = src_t[QW + "_scale"]
    del tensors[f"{MTP}.gate_up_proj"], tensors[f"{MTP}.down_proj"]
    pieces = {k: [] for k in ("gu", "gus", "gu2", "d", "ds", "d2")}
    for e in range(2):
        for proj, keys in (("gate", ("gu", "gus", "gu2")), ("up", ("gu", "gus", "gu2")), ("down", ("d", "ds", "d2"))):
            for key, part in zip(keys, ("weight", "weight_scale", "weight_scale_2")):
                pieces[key].append(src_t[f"{MTP}.{e}.{proj}_proj.{part}"][2])
    j = {k: b"".join(v) for k, v in pieces.items()}
    tensors[f"{MTP}.gate_up_proj"] = ("U8", [2, 32, 16], j["gu"])
    tensors[f"{MTP}.gate_up_proj_scale"] = ("F8_E4M3", [2, 32, 2], j["gus"])
    tensors[f"{MTP}.gate_up_proj_scale_2"] = ("F32", [2, 2], j["gu2"])
    tensors[f"{MTP}.down_proj"] = ("U8", [2, 32, 8], j["d"])
    tensors[f"{MTP}.down_proj_scale"] = ("F8_E4M3", [2, 32, 1], j["ds"])
    tensors[f"{MTP}.down_proj_scale_2"] = ("F32", [2], j["d2"])
    write_out(out, tensors)
    (out / "CONVERSION.json").write_text(json.dumps({"format": "s3"}))
    return src, out, cur, tensors


def write_out(out: Path, tensors: dict) -> None:
    write_safetensors(out / "tensors/m-1.safetensors", tensors)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "tensors/m-1.safetensors" for n in tensors}}))


class ConvCheckS3(unittest.TestCase):
    """tasks/fp4local-spec.md S3: the kept MXFP8 pairs and NVFP4 MTP pieces byte-equal to FP4L, the rest as S2."""

    run_check = ConvCheck.run_check

    def test_a_correct_s3_conversion_passes(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, cur, _ = conversion_triplet_s3(Path(d))
            for jobs in (1, 2):
                rc, rep = self.run_check(src, out, cur, jobs)
                self.assertEqual((rc, rep["failures"], rep["format"]), (0, 0, "s3"), rep)
                r = rep["rules"]
                self.assertEqual({k: r[k].get("tensors", r[k].get("expert_slices")) for k in r},
                                 {"copy": 4, "mxfp8_kept": 1, "csf": 1, "norm": 1, "mtp_nvfp4": 4})
                self.assertEqual(r["mxfp8_kept"]["elements"], 128)
                self.assertEqual(r["mtp_nvfp4"]["elements"], 2 * 3 * 16 * 32)
                self.assertEqual(rep["layout"]["s3_derived"], 1 + 2, "q_proj; MTP gate_up / down")
                self.assertEqual(rep["output_tensors"], 14)

    def test_each_kept_piece_detects_a_one_byte_mismatch(self):
        cases = {QW: "mxfp8_kept", QW + "_scale": "mxfp8_kept", f"{MTP}.gate_up_proj": "mtp_nvfp4",
                 f"{MTP}.gate_up_proj_scale": "mtp_nvfp4", f"{MTP}.gate_up_proj_scale_2": "mtp_nvfp4",
                 f"{MTP}.down_proj": "mtp_nvfp4", f"{MTP}.down_proj_scale": "mtp_nvfp4",
                 f"{MTP}.down_proj_scale_2": "mtp_nvfp4", "lm_head.weight": "copy"}
        for name, rule in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                src, out, cur, tensors = conversion_triplet_s3(Path(d))
                dt, sh, b = tensors[name]
                # The last byte: in the fused tensors that is expert 1, whose slice must be the one flagged.
                tensors[name] = (dt, sh, b[:-1] + bytes([b[-1] ^ 0x01]))
                write_out(out, tensors)
                rc, rep = self.run_check(src, out, cur)
                self.assertEqual((rc, rep["failures"], rep["rules"][rule]["mismatched"]), (1, 1, 1), rep)

    def test_non_canonical_mxfp8_is_refused_even_when_byte_equal(self):
        for what, idx, byte in (("scale 9", 0, 9), ("scale 247", 1, 247), ("NaN weight", None, 0x7F)):
            with self.subTest(what), tempfile.TemporaryDirectory() as d:
                src, out, cur, tensors = conversion_triplet_s3(Path(d))
                srcs = read_tensors(src / "tensors/m-1.safetensors")
                name = QW if idx is None else QW + "_scale"
                dt, sh, b = tensors[name]
                pos = 3 if idx is None else idx
                b = b[:pos] + bytes([byte]) + b[pos + 1:]
                srcs[name] = tensors[name] = (dt, sh, b)
                write_safetensors(src / "tensors/m-1.safetensors", srcs)
                write_out(out, tensors)
                rc, rep = self.run_check(src, out, cur)
                self.assertEqual((rc, rep["rules"]["mxfp8_kept"]["mismatched"]), (1, 1), rep)

    def test_format_comes_from_conversion_json(self):
        with tempfile.TemporaryDirectory() as d:
            src, out, cur, _ = conversion_triplet_s3(Path(d))
            (out / "CONVERSION.json").unlink()
            rc, rep = self.run_check(src, out, cur)
            self.assertEqual((rc, rep["format"]), (1, "s2"), "an S3 layout is not a valid S2 output")
            self.assertGreater(rep["unmapped_outputs"], 0)
        with tempfile.TemporaryDirectory() as d:
            src, out, cur, _, _ = conversion_triplet(Path(d))
            (out / "CONVERSION.json").write_text(json.dumps({"format": "s3"}))
            rc, rep = self.run_check(src, out, cur)
            self.assertEqual((rc, rep["format"]), (1, "s3"), "an S2 layout is not a valid S3 output")
            (out / "CONVERSION.json").write_text(json.dumps({"format": "s9"}))
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ci.main(["conv-check", str(src), str(out), str(cur)]), 2)
            self.assertIn("format", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
