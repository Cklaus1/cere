"""SelfTune ST4 S0/S1 (tasks/selftune-spec.md, ST4 S0 source pin): scripts/native/model_source.py refuses a
checkpoint directory whose file set or hashes differ from the pinned source (DICE-8B's
tests/fixtures/model_configs/dice-8b/source.json).
Hermetic: the directories are built in a temp dir from a synthetic pin; the committed pin is checked for its own
rules and for the file set the maintainer approved (S0 review, 2026-11-12)."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/native"))
sys.path.insert(0, str(ROOT / "scripts"))
from model_inventory import parse_tree  # noqa: E402
from model_source import (SourceError, load_derived, load_source, lfs_pointer, main, tree_bytes, tree_path,  # noqa: E402
                          verify)

DICE_SOURCE = ROOT / "tests/fixtures/model_configs/dice-8b/source.json"
DICE_DERIVED = ROOT / "tests/fixtures/model_configs/dice-8b/derived.json"
INKLING_SOURCE = ROOT / "tests/fixtures/model_configs/inkling-2-mini/source.json"
INKLING_MIRROR = Path("/var/lib/cere/models/Inkling-2-Mini@b85c3cfa")
# inkling design note section 2: the hub revision and the reference decode loop (transformers 046bdb8c, "Add
# Inkling 2 Mini (#51050)", released in 5.24.0).
INKLING_REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
INKLING_LOOP = ("huggingface/transformers", "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3",
                "src/transformers/models/inkling2_loop/modeling_inkling2_loop.py",
                "1c8e65b45a11c13ecad86c088105acd5e4567123ef6599844a95912d242e9f28")
# D-m (maintainer, 2026-11-12): the tokenizer.json the S3d artifact was prepared with, converted in the S2 reference
# image (scripts/native/dice_reference.py tokenizer); a different file needs a new artifact identity.
DICE_TOKENIZER_SHA256 = "0dbef892a2c52e7074ca85c5294c1c48293a93b646903f03f46282d0ddc86d46"
DICE_REFERENCE_IMAGE = "sha256:37e8cf2aa1baaaf0900247899eabeebc31c6c1a0b3f841ef3888d3ae719bed37"

# The maintainer's approved execution set (S0 review, 2026-11-12): the only hub code a reference may import, and the
# authors' decode loop (github.com/deadlykitten4/DICE @1abe8876, evaluation/scripts/sdar_utils.py), run verbatim.
APPROVED_CODE = {
    "configuration_sdar.py": "73f6e1eaa4018f2f4023c4c9f90a4a401f67762a106644e248c35f0233ef5db4",
    "modeling_sdar.py": "6ebf3124a5fd17171be0d2f6f7674eb76dcf997a8a6cae8720131187bb30ac95",
}
APPROVED_LOOP = ("deadlykitten4/DICE", "1abe88761f7715f82da84c838b387ea282b8b898", "evaluation/scripts/sdar_utils.py",
                 "81f549b19c2799cf24235d2baa395630718d2fc607c297c02cdf36c1525db867")

CONTENT = {
    "config.json": (b'{"model_type": "sdar"}\n', "config"),
    "modeling_sdar.py": (b"# model\n", "code"),
    "configuration_sdar.py": (b"# config\n", "code"),
    "tokenization_qwen2.py": (b"# tokenizer\n", "never_execute"),
    "vocab.json": (b"{}\n", "tokenizer"),
    "README.md": (b"readme\n", "doc"),
    "model-00001-of-00001.safetensors": (b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ", "weights"),
}
LOOP = b"def block_diffusion_generate():\n    pass\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_pin() -> dict:
    files = {n: {"sha256": sha(b), "bytes": len(b), "role": r} for n, (b, r) in CONTENT.items()}
    return {"repo": "x/y", "revision": "0" * 40, "sha256": files["config.json"]["sha256"], "files": files,
            "decode_loop": {"repo": "x/z", "commit": "1" * 40, "path": "s/sdar_utils.py", "file": "sdar_utils.py",
                            "sha256": sha(LOOP), "bytes": len(LOOP)}}


class ModelSource(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.pin = synthetic_pin()

    def make(self, mode: str) -> Path:
        d = self.tmp / mode
        d.mkdir()
        for name, (data, role) in CONTENT.items():
            if mode == "execute" and role == "never_execute":
                continue
            (d / name).write_bytes(data)
        if mode == "execute":
            (d / "sdar_utils.py").write_bytes(LOOP)
        return d

    def refused(self, d: Path, mode: str, needle: str, **kw):
        with self.assertRaises(SourceError) as cm:
            verify(self.pin, d, mode, **kw)
        self.assertIn(needle, str(cm.exception))

    def write_pin(self, pin: dict) -> Path:
        p = self.tmp / "source.json"
        p.write_text(json.dumps(pin))
        return p

    def test_exact_directories_pass(self):
        self.assertEqual(verify(self.pin, self.make("download"), "download", hash_weights=True), 7)
        self.assertEqual(verify(self.pin, self.make("execute"), "execute", hash_weights=True), 7)

    def test_an_unpinned_py_is_refused(self):
        d = self.make("execute")
        (d / "tokenization_qwen2_fast.py").write_bytes(b"# fast\n")
        self.refused(d, "execute", "tokenization_qwen2_fast.py is an unpinned remote code")

    def test_a_never_execute_file_is_refused_where_code_runs(self):
        d = self.make("execute")
        (d / "tokenization_qwen2.py").write_bytes(CONTENT["tokenization_qwen2.py"][0])
        self.refused(d, "execute", "tokenization_qwen2.py is an unpinned remote code (execute mode)")

    def test_the_decode_loop_is_required_where_code_runs_and_refused_in_the_mirror(self):
        d = self.make("execute")
        (d / "sdar_utils.py").unlink()
        self.refused(d, "execute", "sdar_utils.py is missing (execute mode)")
        m = self.make("download")
        (m / "sdar_utils.py").write_bytes(LOOP)
        self.refused(m, "download", "sdar_utils.py is an unpinned remote code (download mode)")

    def test_a_changed_byte_is_refused(self):
        for name in ("modeling_sdar.py", "config.json", "sdar_utils.py"):
            with self.subTest(name=name):
                d = self.make("execute") if name == "sdar_utils.py" else self.make("download")
                p = d / name
                data = bytearray(p.read_bytes())
                data[0] ^= 1
                p.write_bytes(bytes(data))
                self.refused(d, d.name, f"{name} sha256")
                d.rename(self.tmp / f"done-{name}")

    def test_a_missing_or_extra_file_is_refused(self):
        d = self.make("download")
        (d / "vocab.json").unlink()
        self.refused(d, "download", "vocab.json is missing")
        (d / "vocab.json").write_bytes(CONTENT["vocab.json"][0])
        (d / "tokenizer.json").write_bytes(b"{}")
        self.refused(d, "download", "tokenizer.json is an unpinned file")

    def test_weights_by_size_unless_hashed(self):
        d = self.make("download")
        p = d / "model-00001-of-00001.safetensors"
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(bytes(data))
        self.assertEqual(verify(self.pin, d, "download"), 7)
        self.refused(d, "download", "model-00001-of-00001.safetensors sha256", hash_weights=True)
        p.write_bytes(bytes(data) + b" ")
        self.refused(d, "download", "has 17 bytes, the pin 16")

    def test_a_symlink_is_refused(self):
        d = self.make("execute")
        (d / "modeling_sdar.py").unlink()
        (d / "modeling_sdar.py").symlink_to(self.tmp / "elsewhere.py")
        self.refused(d, "execute", "modeling_sdar.py is not a regular file")

    def test_malformed_pins_are_refused(self):
        cases = [
            (lambda p: p["files"]["modeling_sdar.py"].update(role="doc"), "does not match its kind"),
            (lambda p: p["files"]["README.md"].update(role="code"), "does not match its kind"),
            (lambda p: p["files"]["README.md"].update(role="script"), "is not one of"),
            (lambda p: p["files"]["README.md"].update(sha256="A" * 64), "64 lowercase hex"),
            (lambda p: p["files"]["README.md"].update(bytes=0), "positive integer"),
            (lambda p: p.update(sha256="0" * 64), "config.json sha256 differs"),
            (lambda p: p["files"].update({"../x": p["files"].pop("README.md")}), "plain file name"),
            (lambda p: p.pop("decode_loop"), "decode_loop is absent"),
            (lambda p: p["decode_loop"].update(file="modeling_sdar.py"), "apart from files"),
        ]
        for mutate, needle in cases:
            with self.subTest(needle=needle):
                pin = copy.deepcopy(self.pin)
                mutate(pin)
                with self.assertRaises(SourceError) as cm:
                    load_source(self.write_pin(pin))
                self.assertIn(needle, str(cm.exception))

    def test_cli_exit_codes(self):
        p = self.write_pin(self.pin)
        d = self.make("download")
        self.assertEqual(main([str(p), str(d), "--hash-weights"]), 0)
        (d / "extra.py").write_bytes(b"")
        self.assertEqual(main([str(p), str(d)]), 1)

    def prepared(self, d: Path | None = None) -> Path:
        d = d or self.make("prepare")
        self.assertEqual(verify(self.pin, d, "prepare", write_tree=True), 7)
        return d

    def test_prepare_mode_writes_and_checks_the_tree_cerec_reads(self):
        d = self.prepared()
        tree = tree_path(self.pin, d)
        self.assertEqual(tree, d / ".cache/huggingface/trees" / ("0" * 40 + ".json"))
        self.assertEqual(verify(self.pin, d, "prepare", hash_weights=True), 7)
        # The tree parses under the inventory's (and cerec's) rules: git blob ids, LFS only for weights.
        parsed = parse_tree(tree.name, tree.read_bytes())
        self.assertEqual(set(parsed["files"]), set(CONTENT))
        weights = "model-00001-of-00001.safetensors"
        entry = self.pin["files"][weights]
        pointer = lfs_pointer(entry)
        self.assertEqual(parsed["files"][weights]["blob_id"],
                         hashlib.sha1(b"blob %d\0" % len(pointer) + pointer).hexdigest())
        self.assertEqual(parsed["files"][weights]["lfs_sha256"], entry["sha256"])
        data = CONTENT["vocab.json"][0]
        self.assertEqual(parsed["files"]["vocab.json"]["blob_id"],
                         hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
        self.assertIsNone(parsed["files"]["vocab.json"].get("lfs_sha256"))
        # The mirror modes still refuse the tree directory.
        self.refused(d, "download", ".cache")

    def test_prepare_mode_refuses_a_missing_changed_or_crowded_tree(self):
        d = self.make("prepare")
        self.refused(d, "prepare", "is not a directory (prepare mode)")
        self.prepared(d)
        tree = tree_path(self.pin, d)
        tree.write_bytes(tree.read_bytes().replace(b'"size": 3', b'"size": 4'))
        self.refused(d, "prepare", "differs from the tree rendered from the pin")
        # write_tree never overwrites an existing tree.
        self.refused(d, "prepare", "differs from the tree rendered from the pin", write_tree=True)
        tree.write_bytes(tree_bytes(self.pin, d))
        (tree.parent / ("1" * 40 + ".json")).write_bytes(b"{}")
        self.refused(d, "prepare", "not ['" + "0" * 40 + ".json']")

    def test_write_tree_needs_prepare_mode_and_a_hex_revision(self):
        self.refused(self.make("download"), "download", "--write-tree needs prepare mode", write_tree=True)
        self.pin["revision"] = "main"
        self.refused(self.make("prepare"), "prepare", "not 40 lowercase hex digits", write_tree=True)

    def derived(self, data: bytes = b'{"model": {}}\n') -> dict:
        return {"tokenizer.json": {"sha256": sha(data), "bytes": len(data), "role": "tokenizer"}}

    def test_prepare_mode_admits_only_the_recorded_derived_file(self):
        """D-m bar (b): a tokenizer.json is admitted only with the recorded size and sha256."""
        d = self.prepared()
        data = b'{"model": {}}\n'
        (d / "tokenizer.json").write_bytes(data)
        self.refused(d, "prepare", "tokenizer.json is an unpinned file (prepare mode)")
        self.assertEqual(verify(self.pin, d, "prepare", derived=self.derived(data)), 8)
        other = b'{"model": []}\n'
        (d / "tokenizer.json").write_bytes(other)
        self.refused(d, "prepare", "tokenizer.json sha256 " + sha(other) + " differs from the derived record",
                     derived=self.derived(data))
        (d / "tokenizer.json").unlink()
        self.refused(d, "prepare", "tokenizer.json is missing (prepare mode)", derived=self.derived(data))

    def test_derived_files_need_prepare_mode_and_a_new_name(self):
        self.refused(self.make("download"), "download", "derived files need prepare mode", derived=self.derived())
        clash = {"vocab.json": dict(self.pin["files"]["vocab.json"])}
        self.refused(self.prepared(), "prepare", "are pinned source files", derived=clash)

    def test_malformed_derived_records_are_refused(self):
        cases = [({"files": {}}, "files is absent or empty"),
                 ({"files": {"x.py": self.derived()["tokenizer.json"]}}, "plain data file name"),
                 ({"files": {"t.json": dict(self.derived()["tokenizer.json"], role="weights")}}, "is not config"),
                 ({"files": {"t.json": dict(self.derived()["tokenizer.json"], bytes=0)}}, "positive integer")]
        for doc, needle in cases:
            with self.subTest(needle=needle):
                p = self.tmp / "derived.json"
                p.write_text(json.dumps(doc))
                with self.assertRaises(SourceError) as cm:
                    load_derived(p)
                self.assertIn(needle, str(cm.exception))

    def subdir_pin(self) -> tuple[dict, bytes]:
        """The synthetic pin plus `tiktoken/tokenizer.model`, an LFS tokenizer file (Inkling 2 Mini's layout)."""
        data = b"tiktoken ranks\n"
        pin = copy.deepcopy(self.pin)
        pin["files"]["tiktoken/tokenizer.model"] = {"sha256": sha(data), "bytes": len(data), "role": "tokenizer",
                                                   "lfs": True}
        self.pin = pin
        return pin, data

    def test_a_pinned_subdirectory_passes_in_every_mode(self):
        _, data = self.subdir_pin()
        self.assertEqual(load_source(self.write_pin(self.pin))["files"]["tiktoken/tokenizer.model"]["lfs"], True)
        for mode in ("download", "execute", "prepare"):
            with self.subTest(mode=mode):
                d = self.make(mode)
                (d / "tiktoken").mkdir()
                (d / "tiktoken/tokenizer.model").write_bytes(data)
                self.assertEqual(verify(self.pin, d, mode, write_tree=mode == "prepare"), 8)
        # the tree names the file by its path, as an LFS file: the pointer's blob id and the oid
        parsed = parse_tree("0" * 40 + ".json", tree_path(self.pin, d).read_bytes())
        entry = parsed["files"]["tiktoken/tokenizer.model"]
        pointer = lfs_pointer(self.pin["files"]["tiktoken/tokenizer.model"])
        self.assertEqual(entry["blob_id"], hashlib.sha1(b"blob %d\0" % len(pointer) + pointer).hexdigest())
        self.assertEqual(entry["lfs_sha256"], sha(data))

    def test_a_subdirectory_differing_from_the_pin_is_refused(self):
        _, data = self.subdir_pin()
        d = self.make("download")
        self.refused(d, "download", "tiktoken/tokenizer.model is missing")
        (d / "tiktoken").mkdir()
        (d / "tiktoken/tokenizer.model").write_bytes(data + b"x")
        self.refused(d, "download", "tiktoken/tokenizer.model has 16 bytes, the pin 15")
        (d / "tiktoken/tokenizer.model").write_bytes(bytes([data[0] ^ 1]) + data[1:])
        self.refused(d, "download", "tiktoken/tokenizer.model sha256")
        (d / "tiktoken/tokenizer.model").write_bytes(data)
        (d / "tiktoken/extra.py").write_bytes(b"")
        self.refused(d, "download", "tiktoken/extra.py is an unpinned remote code")
        (d / "tiktoken/extra.py").unlink()
        (d / "tiktoken/nested").mkdir()
        self.refused(d, "download", "tiktoken/nested is an unpinned file")
        (d / "tiktoken/nested").rmdir()
        (d / "other").mkdir()
        self.refused(d, "download", "other is an unpinned file")

    def test_an_unpinned_or_linked_subdirectory_is_refused(self):
        d = self.make("download")
        (d / "tiktoken").mkdir()
        self.refused(d, "download", "tiktoken/ is an unpinned directory")
        self.subdir_pin()
        (d / "tiktoken").rmdir()
        (self.tmp / "real").mkdir()
        (d / "tiktoken").symlink_to(self.tmp / "real")
        self.refused(d, "download", "tiktoken is an unpinned file")

    def test_malformed_subdirectory_and_lfs_pins_are_refused(self):
        self.subdir_pin()
        cases = [
            (lambda p: p["files"].update({"other/t.model": p["files"].pop("tiktoken/tokenizer.model")}),
             "not a plain file name"),
            (lambda p: p["files"].update({"tiktoken/a/t.model": p["files"].pop("tiktoken/tokenizer.model")}),
             "not a plain file name"),
            (lambda p: p["files"].update({"tiktoken/..": p["files"].pop("tiktoken/tokenizer.model")}),
             "not a plain file name"),
            (lambda p: p["files"].update({"tiktoken/": p["files"].pop("tiktoken/tokenizer.model")}),
             "not a plain file name"),
            (lambda p: p["files"]["tiktoken/tokenizer.model"].update(lfs=False), "lfs is only `true`"),
            (lambda p: p["files"]["model-00001-of-00001.safetensors"].update(lfs=True), "lfs is only `true`"),
            (lambda p: p["files"]["README.md"].update(extra=1), "an entry is exactly"),
        ]
        for mutate, needle in cases:
            with self.subTest(needle=needle):
                pin = copy.deepcopy(self.pin)
                mutate(pin)
                with self.assertRaises(SourceError) as cm:
                    load_source(self.write_pin(pin))
                self.assertIn(needle, str(cm.exception))


class DicePin(unittest.TestCase):
    """The committed DICE-8B pin holds the maintainer-approved set and the decode loop, by sha256."""

    def setUp(self):
        self.source = load_source(DICE_SOURCE)

    def test_code_set_is_the_approved_one(self):
        code = {n: e["sha256"] for n, e in self.source["files"].items() if e["role"] == "code"}
        self.assertEqual(code, APPROVED_CODE)
        never = sorted(n for n, e in self.source["files"].items() if e["role"] == "never_execute")
        self.assertEqual(never, ["tokenization_qwen2.py", "tokenization_qwen2_fast.py"])

    def test_decode_loop_pin(self):
        loop = self.source["decode_loop"]
        self.assertEqual((loop["repo"], loop["commit"], loop["path"], loop["sha256"]), APPROVED_LOOP)

    def test_weights_are_safetensors_only(self):
        weights = sorted(n for n, e in self.source["files"].items() if e["role"] == "weights")
        self.assertEqual(weights, [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)])
        self.assertFalse([n for n in self.source["files"] if n.endswith((".bin", ".pt", ".pth", ".pkl"))])

    def test_derived_tokenizer_is_the_recorded_conversion(self):
        doc = json.loads(DICE_DERIVED.read_text())
        self.assertEqual(load_derived(DICE_DERIVED), {"tokenizer.json": {
            "sha256": DICE_TOKENIZER_SHA256, "bytes": 11422833, "role": "tokenizer"}})
        self.assertEqual(doc["image_id"], DICE_REFERENCE_IMAGE)
        self.assertEqual(doc["source_json_sha256"], sha(DICE_SOURCE.read_bytes()))
        prompts = ROOT / "tests/fixtures/dice8b_reference/prompts.json"
        self.assertEqual(doc["prompts_sha256"], sha(prompts.read_bytes()))
        self.assertEqual(doc["prompts_equal"], 64)


class InklingPin(unittest.TestCase):
    """tasks/inkling-spec.md S0 / G0: the committed Inkling 2 Mini pin: no remote code, the reference decode loop at
    transformers 046bdb8c, `tiktoken/` the only subdirectory, and (where the mirror is readable) the mirror equal to
    it in download mode."""

    def setUp(self):
        self.source = load_source(INKLING_SOURCE)

    def test_revision_and_decode_loop(self):
        self.assertEqual(self.source["revision"], INKLING_REVISION)
        loop = self.source["decode_loop"]
        self.assertEqual((loop["repo"], loop["commit"], loop["path"], loop["sha256"]), INKLING_LOOP)
        self.assertEqual(loop["file"], "modeling_inkling2_loop.py")

    def test_file_set(self):
        files = self.source["files"]
        self.assertFalse([n for n, e in files.items() if e["role"] in ("code", "never_execute")])
        self.assertEqual(sorted(n for n, e in files.items() if e["role"] == "weights"),
                         ["model-00001-of-00001.safetensors", "mtp.safetensors"])
        self.assertEqual(sorted(n for n in files if "/" in n), ["tiktoken/tokenizer.model"])
        self.assertEqual(sorted(n for n, e in files.items() if e.get("lfs")),
                         ["tiktoken/tokenizer.model", "tokenizer.json"])
        fixture = INKLING_SOURCE.parent
        for name in ("config.json", "hf_quant_config.json"):
            self.assertEqual(sha((fixture / name).read_bytes()), files[name]["sha256"], name)

    @unittest.skipUnless(os.access(INKLING_MIRROR, os.R_OK), "the Inkling 2 Mini mirror is not readable here")
    def test_mirror_verifies_in_download_mode(self):
        """G0 (weights by size here; `model_source.py ... --hash-weights` hashes them too)."""
        self.assertEqual(verify(self.source, INKLING_MIRROR, "download"), len(self.source["files"]))


if __name__ == "__main__":
    unittest.main()
