"""M2.1 B2/B3: v2 prepared artifacts, the shared invalid table and the reference inspector."""
from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from artifact_helpers import (EXAMPLE, MODEL_FAMILIES, ROOT, TABLE, copy_example, inventory_case_src,  # noqa: E402
                              prepare_case)
from cere_contracts import (ALL_ERROR_CLASSES, ERROR_CLASSES, ArtifactError, limits_with, load_json, validate,  # noqa: E402
                            parse_json)
from model_inventory import FileIO, inventory, parse_header  # noqa: E402
import token_map  # noqa: E402
import verify_artifact  # noqa: E402
from verify_artifact import inspect_artifact, stamp_facts, stamp_path  # noqa: E402

TABLE_DATA = load_json(TABLE)
REAL_TOKEN_MAP = Path("/var/lib/cere/models/qwen38-flash-next/flash-next-64k.pt")  # read-only, metadata-sized
REAL_VOCAB = 248320  # m21 design note: text_config.vocab_size of the real model
CASES = TABLE_DATA["cases"]


def header_file(case) -> bytes:
    head = case["header"].encode() if "header" in case else bytes.fromhex(case["header_hex"])
    n = case.get("n", len(head))
    return struct.pack("<Q", n) + head + bytes(case.get("data_len", 0))


def parse_header_case(case) -> dict:
    """The header case through the same bounds-then-parse path the inspector uses."""
    data = header_file(case)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.safetensors"
        p.write_bytes(data)
        io = FileIO()
        fd = os.open(p, os.O_RDONLY)
        try:
            limits = limits_with(case.get("limits"))
            prefix = verify_artifact.read_header_prefix(io, fd, len(data), limits)
            return parse_header(prefix, len(data), limits)
        finally:
            os.close(fd)


class TableTests(unittest.TestCase):
    def test_table_is_generated(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/fixtures/make_artifact_invalid.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_table_shape(self):
        self.assertGreaterEqual(len(CASES), 36)
        self.assertEqual(TABLE_DATA["error_classes"], [c for t in ("header", "artifact", "token_map", "inventory")
                                                       for c in ERROR_CLASSES[t]])
        self.assertEqual(TABLE_DATA["default_limits"], limits_with(None))
        ids = [c["id"] for c in CASES]
        self.assertEqual(len(ids), len(set(ids)))
        for c in CASES:
            self.assertIn(c["tag"], ("header", "artifact", "token_map", "inventory"))
            self.assertIn(c["error"], ERROR_CLASSES["header"] + ERROR_CLASSES["artifact"]
                          if c["tag"] == "artifact" else ERROR_CLASSES[c["tag"]])

    def test_table_covers_every_listed_kind(self):
        """Every Q15 bound and every kind named in the B2 acceptance cell has a case."""
        by = {}
        for c in CASES:
            by.setdefault(c["error"], []).append(c)
        ids = {c["id"] for c in CASES}
        limits = {k for c in CASES for k in c.get("limits", {})}
        self.assertEqual(limits, set(limits_with(None)))  # every Q15 bound is exercised
        for cls in ("dtype", "family_shape", "group_nvfp4", "group_ple_table", "group_ple_module", "family_unmatched",
                    "family_ambiguous", "family_totals", "header_hash", "unlisted_file", "symlink", "special_file",
                    "unsafe_path", "manifest_size", "manifest_noncanonical", "token_range", "token_order",
                    "token_empty", "pickle_grammar", "zip_unsupported", "zip_entry"):
            self.assertIn(cls, by, cls)
        for case_id in ("header_dtype_unknown", "artifact_scalar_shape_misuse", "artifact_manifest_reindented",
                        "artifact_manifest_reordered_keys", "artifact_path_traversal", "artifact_ple_module_missing",
                        "token_map_global_opcode", "token_map_id_at_vocab", "token_map_negative_binint",
                        "token_map_decreasing", "token_map_empty_list", "token_map_trailing_after_stop",
                        "token_map_deflated_entry", "token_map_zip64", "token_map_two_data_pkl"):
            self.assertIn(case_id, ids)
        self.assertEqual({c["error"] for c in CASES} - ALL_ERROR_CLASSES, set())


class HeaderCaseTests(unittest.TestCase):
    def test_every_header_case_rejected_with_its_class(self):
        for c in (c for c in CASES if c["tag"] == "header"):
            with self.subTest(case=c["id"]):
                with self.assertRaises(ArtifactError) as cm:
                    parse_header_case(c)
                self.assertEqual(cm.exception.cls, c["error"], str(cm.exception))

    def test_header_cli_rejects_with_class_on_first_stderr_line(self):
        c = next(c for c in CASES if c["tag"] == "header" and c["error"] == "duplicate_key")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.safetensors"
            p.write_bytes(header_file(c))
            r = subprocess.run([sys.executable, str(ROOT / "scripts/model_inventory.py"), "--dump-header", str(p)],
                               capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertTrue(r.stderr.splitlines()[0].startswith("duplicate_key: "), r.stderr)


class TokenMapCaseTests(unittest.TestCase):
    def test_every_token_map_case_rejected_with_its_class(self):
        for c in (c for c in CASES if c["tag"] == "token_map"):
            with self.subTest(case=c["id"]):
                with self.assertRaises(ArtifactError) as cm:
                    token_map.convert(bytes.fromhex(c["file_hex"]), c["vocab_size"], limits=c.get("limits"))
                self.assertEqual(cm.exception.cls, c["error"], str(cm.exception))

    def test_file_over_pkl_limit_plus_slack_rejected_before_parsing(self):
        """cerec checks `max_token_map_pkl_bytes + 1 MiB` against fstat before reading; the reference agrees."""
        limits = {"max_token_map_pkl_bytes": 8}
        with self.assertRaises(ArtifactError) as cm:
            token_map.convert(bytes(8 + (1 << 20) + 1), 10, limits=limits)
        self.assertEqual(cm.exception.cls, "zip_entry")
        with self.assertRaises(ArtifactError) as cm:
            token_map.convert(bytes(8 + (1 << 20)), 10, limits=limits)
        self.assertEqual(cm.exception.cls, "zip_structure")

    def test_valid_token_maps_convert(self):
        """PyTorch's real layout (FB-padded local headers, zip64 end record + locator) is accepted."""
        self.assertTrue(TABLE_DATA["valid_token_maps"])
        for c in TABLE_DATA["valid_token_maps"]:
            with self.subTest(case=c["id"]):
                out = token_map.convert(bytes.fromhex(c["file_hex"]), c["vocab_size"])
                self.assertEqual(out.hex(), c["u32le_hex"])

    # os.path.isfile swallows EACCES (a non-root CI runner cannot stat /root); Path.is_file re-raises it on 3.11.
    @unittest.skipUnless(os.path.isfile(REAL_TOKEN_MAP) and os.access(REAL_TOKEN_MAP, os.R_OK),
                         f"{REAL_TOKEN_MAP} absent or unreadable")
    def test_real_token_map_converts(self):
        out = token_map.convert(REAL_TOKEN_MAP.read_bytes(), REAL_VOCAB)
        ids = struct.unpack(f"<{len(out) // 4}I", out)
        self.assertEqual(len(ids), 65536)
        self.assertTrue(all(a < b for a, b in zip(ids, ids[1:])))
        self.assertLess(ids[-1], REAL_VOCAB)

    def test_token_map_cli_rejects_every_case(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            for c in (c for c in CASES if c["tag"] == "token_map"):
                with self.subTest(case=c["id"]):
                    pt = Path(d) / "map.pt"
                    pt.write_bytes(bytes.fromhex(c["file_hex"]))
                    cfg.write_text(json.dumps({"text_config": {"vocab_size": c["vocab_size"]}}))
                    cmd = [sys.executable, str(ROOT / "scripts/token_map.py"), str(pt), "--config", str(cfg)]
                    if "limits" in c:
                        cmd += ["--limits", json.dumps(c["limits"])]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual((r.returncode, r.stdout), (2, ""))
                    self.assertTrue(r.stderr.startswith(c["error"] + ": "), r.stderr)


class InventoryCaseTests(unittest.TestCase):
    def test_every_inventory_case_rejected_with_its_class(self):
        cases = [c for c in CASES if c["tag"] == "inventory"]
        self.assertTrue(cases)
        for c in cases:
            with self.subTest(case=c["id"]), tempfile.TemporaryDirectory() as d:
                src = Path(d) / "src"
                inventory_case_src(src, c)
                with self.assertRaises(ArtifactError) as cm:
                    inventory(src, MODEL_FAMILIES)
                self.assertEqual(cm.exception.cls, c["error"], str(cm.exception))


class CountingIO(FileIO):
    """Injectable opener: records every pread so tests can count payload vs metadata reads."""

    def __init__(self):
        self.reads = []
        self.opened = []

    def open(self, name, flags, dir_fd=None):
        self.opened.append((name, flags))
        return super().open(name, flags, dir_fd=dir_fd)

    def pread(self, fd, n, offset):
        data = super().pread(fd, n, offset)
        self.reads.append((fd, offset, len(data)))
        return data


class ArtifactCaseTests(unittest.TestCase):
    """Every `artifact` case of the shared table, on a copy of the committed example (cerec iterates the same)."""

    def test_positive_example_passes(self):
        digest = (EXAMPLE / "manifest.sha256").read_text().strip()
        self.assertEqual(digest, hashlib.sha256((EXAMPLE / "manifest.json").read_bytes()).hexdigest())
        r = inspect_artifact(EXAMPLE, digest, mode="full")
        self.assertEqual(r["outcome"], "verified")

    def test_every_artifact_case_rejected_with_its_class(self):
        for c in (c for c in CASES if c["tag"] == "artifact"):
            with self.subTest(case=c["id"]), tempfile.TemporaryDirectory() as d:
                p = Path(d) / "a"
                digest = prepare_case(p, c)
                with self.assertRaises(ArtifactError) as cm:
                    inspect_artifact(p, digest, mode="full", limits=c.get("limits"))
                self.assertEqual(cm.exception.cls, c["error"], str(cm.exception))


class ExampleContractTests(unittest.TestCase):
    """B2 positive: the committed v2 example passes schema v2 + semantic rules and agrees with the fixture configs."""

    def setUp(self):
        self.m = load_json(EXAMPLE / "manifest.json")
        self.fx = ROOT / "tests/fixtures"

    def test_example_manifest_validates_schema_and_rules(self):
        self.assertEqual(self.m["schema_version"], "2")
        validate("execution_artifact", self.m)

    def test_embedded_profile_and_rules_equal_fixture_configs(self):
        self.assertEqual(self.m["family_rules"], load_json(self.fx / "model_src.families.json"))
        self.assertEqual(self.m["profile"], load_json(self.fx / "model_src.profile.json"))
        self.assertEqual(self.m["profile_id"], self.m["profile"]["profile_id"])

    def test_hashes_equal_lock_and_families_equal_inventory(self):
        lock = load_json(self.fx / "model_src.lock.json")
        self.assertEqual((self.m["model_hash"], self.m["tokenizer_hash"], self.m["template_hash"]),
                         (lock["model_content_sha256"], lock["tokenizer_sha256"], lock["template_sha256"]))
        inv = load_json(self.fx / "model_src.inventory.json")
        self.assertEqual(self.m["families"], inv["families"])
        self.assertEqual(self.m["source"], inv["source"])

    def test_files_listed_exactly_once_and_match_directory(self):
        paths = [f["path"] for f in self.m["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        on_disk = {str(q.relative_to(EXAMPLE)) for q in EXAMPLE.rglob("*") if q.is_file()}
        self.assertEqual(on_disk - {"manifest.json", "manifest.sha256"}, set(paths))


class InspectorModeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.p = Path(tmp.name) / "artifact"
        self.digest = copy_example(self.p)
        self.manifest = load_json(self.p / "manifest.json")

    def payload_total(self):
        return sum(f["bytes"] - f.get("safetensors", {}).get("header_bytes", 0) for f in self.manifest["files"])

    def write_stamp(self, source="full_verify", legacy=False):
        """Format 2 names each file by its fsid; `legacy` writes format 1 (no `format` key, `dev` per file)."""
        files = {}
        for f in self.manifest["files"]:
            st = os.stat(self.p / f["path"])
            fs = {"dev": st.st_dev} if legacy else {"fsid": f"{os.statvfs(self.p / f['path']).f_fsid:016x}"}
            files[f["path"]] = stamp_facts(st) | fs
        stamp = {"manifest_sha256": self.digest, "stamp_source": source, "source_fingerprint": "0" * 64,
                 "files": files, "verified_utc": "2026-10-06T00:00:00Z"}
        if not legacy:
            stamp["format"] = 2
        stamp_path(self.p).write_text(json.dumps(stamp))

    def test_headers_mode_reads_no_payload(self):
        io = CountingIO()
        r = inspect_artifact(self.p, self.digest, mode="headers", io=io)
        self.assertEqual(r["payload_bytes_read"], 0)
        self.assertEqual(r["metadata_bytes_read"], sum(n for _fd, _o, n in io.reads))
        self.assertTrue(all(flags & os.O_NOFOLLOW for _n, flags in io.opened))

    def test_full_mode_reads_every_payload_byte_once(self):
        io = CountingIO()
        r = inspect_artifact(self.p, self.digest, mode="full", io=io)
        self.assertEqual(r["payload_bytes_read"], self.payload_total())
        self.assertEqual(r["payload_bytes_read"] + r["metadata_bytes_read"], sum(n for _fd, _o, n in io.reads))

    def test_stamp_mode_reads_no_payload(self):
        self.write_stamp()
        io = CountingIO()
        r = inspect_artifact(self.p, self.digest, mode="stamp", io=io)
        self.assertEqual((r["outcome"], r["payload_bytes_read"]), ("verified", 0))

    def test_stamp_mode_rejects_prepare_hash_stamp(self):
        self.write_stamp("prepare_hash")
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertEqual(cm.exception.cls, "stamp")
        self.assertIn("full verify required", str(cm.exception))

    def test_stamp_mode_rejects_missing_stamp(self):
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertEqual(cm.exception.cls, "stamp")

    def test_stamp_mode_rejects_after_touch(self):
        self.write_stamp()
        f = self.p / self.manifest["files"][0]["path"]
        st = os.stat(f)
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertIn("full verify required", str(cm.exception))

    def test_stamp_mode_rejects_after_chmod(self):
        self.write_stamp()
        f = self.p / self.manifest["files"][-1]["path"]
        os.chmod(f, 0o400)
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertIn("full verify required", str(cm.exception))

    def test_stamp_mode_rejects_each_identity_field_alone(self):
        """Q7: every recorded fstat field invalidates the stamp on its own. The stamp is rewritten with exactly one
        field changed; ctime and the other fields stay as they are, so no field can hide behind another."""
        path = self.manifest["files"][0]["path"]
        self.write_stamp()
        good = load_json(stamp_path(self.p))
        inspect_artifact(self.p, self.digest, mode="stamp")  # the unchanged stamp passes
        for field in ("fsid", "ino", "size", "mtime_ns", "ctime_ns", "mode"):
            with self.subTest(field=field):
                stamp = json.loads(json.dumps(good))
                rec = stamp["files"][path]
                rec[field] = f"{int(rec[field], 16) ^ 1:016x}" if field == "fsid" else rec[field] + 1
                stamp_path(self.p).write_text(json.dumps(stamp))
                with self.assertRaises(ArtifactError) as cm:
                    inspect_artifact(self.p, self.digest, mode="stamp")
                self.assertEqual(cm.exception.cls, "stamp")
                self.assertIn(f"full verify required: {path} changed", str(cm.exception))

    def test_stamp_mode_old_format_needs_a_device_match(self):
        """Owner 2026-10-15: a format-1 stamp (`dev` per file) passes while st_dev matches; after a renumbering it is
        refused as predating the fs-uuid format."""
        path = self.manifest["files"][0]["path"]
        self.write_stamp(legacy=True)
        inspect_artifact(self.p, self.digest, mode="stamp")
        stamp = load_json(stamp_path(self.p))
        stamp["files"][path]["dev"] += 1
        stamp_path(self.p).write_text(json.dumps(stamp))
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertEqual(cm.exception.cls, "stamp")
        self.assertIn("stamp predates fs-uuid format; full verify required", str(cm.exception))

    def test_stamp_mode_format_2_by_dev_is_refused_on_a_device_change(self):
        """A format-2 file named by `dev` (no UUID-derived id) is a plain change when st_dev differs."""
        path = self.manifest["files"][0]["path"]
        self.write_stamp()
        stamp = load_json(stamp_path(self.p))
        rec = stamp["files"][path]
        del rec["fsid"]
        rec["dev"] = os.stat(self.p / path).st_dev
        stamp_path(self.p).write_text(json.dumps(stamp))
        inspect_artifact(self.p, self.digest, mode="stamp")
        rec["dev"] += 1
        stamp_path(self.p).write_text(json.dumps(stamp))
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertIn(f"full verify required: {path} changed", str(cm.exception))

    def test_full_mode_detects_one_byte_flip_inside_a_tensor(self):
        shard = next(f for f in self.manifest["files"] if f["role"] == "weights")
        p = self.p / shard["path"]
        b = bytearray(p.read_bytes())
        b[shard["safetensors"]["header_bytes"] + 1] ^= 0x01
        p.write_bytes(bytes(b))
        inspect_artifact(self.p, self.digest, mode="headers")  # headers alone cannot see it
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="full")
        self.assertEqual(cm.exception.cls, "file_hash")

    def test_manifest_parsed_from_verified_bytes(self):
        original = (self.p / "manifest.json").read_bytes()
        seen = []

        def parser(b):
            seen.append(b)
            return parse_json(b)
        with mock.patch.object(verify_artifact, "parse_json", side_effect=parser):
            inspect_artifact(self.p, self.digest, mode="headers")
        self.assertEqual(seen, [original])

    def test_full_verify_then_stamp_reads_metadata_only(self):
        io = CountingIO()
        full = inspect_artifact(self.p, self.digest, mode="full", io=io)
        self.assertEqual((full["outcome"], full["payload_bytes_read"]), ("verified", self.payload_total()))
        self.write_stamp()
        io = CountingIO()
        r = inspect_artifact(self.p, self.digest, mode="stamp", io=io)
        self.assertEqual((r["outcome"], r["payload_bytes_read"]), ("verified", 0))
        self.assertEqual(r["metadata_bytes_read"], sum(n for _fd, _o, n in io.reads))
        self.assertLess(r["metadata_bytes_read"], full["payload_bytes_read"] + full["metadata_bytes_read"])

    def test_stamp_mode_rejects_a_record_with_fsid_and_dev(self):
        """A record is exactly the five fstat fields plus `fsid` (format 2) or `dev`, as cerec's stamp parser takes
        it: a record holding both, a format-1 record by `fsid`, or one with an extra key is malformed."""
        path = self.manifest["files"][0]["path"]
        self.write_stamp()
        good = load_json(stamp_path(self.p))
        dev = os.stat(self.p / path).st_dev
        cases = {"fsid_and_dev": ({"dev": dev}, False), "extra_key": ({"extra": 0}, False),
                 "legacy_fsid": ({}, True)}
        for name, (extra, legacy) in cases.items():
            with self.subTest(name):
                stamp = json.loads(json.dumps(good))
                stamp["files"][path] |= extra
                if legacy:
                    del stamp["format"]
                    for p, rec in stamp["files"].items():
                        if p != path:
                            del rec["fsid"]
                            rec["dev"] = os.stat(self.p / p).st_dev
                stamp_path(self.p).write_text(json.dumps(stamp))
                with self.assertRaises(ArtifactError) as cm:
                    inspect_artifact(self.p, self.digest, mode="stamp")
                self.assertEqual(cm.exception.cls, "stamp")
                self.assertIn(f"full verify required: {path} record is malformed", str(cm.exception))

    def test_stamp_mode_rejects_stamp_for_another_manifest(self):
        self.write_stamp()
        stamp = load_json(stamp_path(self.p))
        stamp["manifest_sha256"] = "f" * 64
        stamp_path(self.p).write_text(json.dumps(stamp))
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertEqual(cm.exception.cls, "stamp")
        self.assertIn("another manifest", str(cm.exception))

    def test_stamp_mode_rejects_same_bytes_new_inode(self):
        self.write_stamp()
        f = self.p / self.manifest["files"][0]["path"]
        data, st = f.read_bytes(), os.stat(f)
        f.unlink()
        f.write_bytes(data)
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.chmod(f, st.st_mode)
        with self.assertRaises(ArtifactError) as cm:
            inspect_artifact(self.p, self.digest, mode="stamp")
        self.assertEqual(cm.exception.cls, "stamp")

    def test_cli_full_and_stamp_accept_copy(self):
        script = str(ROOT / "scripts/verify_artifact.py")
        self.write_stamp()
        for mode, payload in (("full", self.payload_total()), ("stamp", 0)):
            r = subprocess.run([sys.executable, script, str(self.p), "--expected-manifest-sha256", self.digest,
                                "--mode", mode], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertEqual((out["outcome"], out["payload_bytes_read"]), ("verified", payload))

    def test_cli_report_and_rejection(self):
        script = str(ROOT / "scripts/verify_artifact.py")
        r = subprocess.run([sys.executable, script, str(self.p), "--expected-manifest-sha256", self.digest,
                            "--mode", "headers"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["payload_bytes_read"], 0)
        r = subprocess.run([sys.executable, script, str(self.p), "--expected-manifest-sha256", "0" * 64,
                            "--mode", "full"], capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertTrue(r.stderr.startswith("manifest_digest: "), r.stderr)


class InspectorWithoutArtifactTests(unittest.TestCase):
    def test_root_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "real").mkdir()
            os.symlink(Path(d) / "real", Path(d) / "link")
            with self.assertRaises(ArtifactError) as cm:
                inspect_artifact(Path(d) / "link", "0" * 64, mode="headers")
            self.assertEqual(cm.exception.cls, "symlink")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            inspect_artifact(ROOT, "0" * 64, mode="lazy")


if __name__ == "__main__":
    unittest.main()
