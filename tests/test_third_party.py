"""M2.2 B3: executor/THIRD_PARTY.json, the PROVENANCE/build.json schemas and scripts/check_third_party.py."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_third_party.py"
FIXTURES = ROOT / "tests/fixtures/native"
IMAGE_ID = "sha256:6af86bea9baebe6c15fe91cb23cc7d33c362943d11c01daaf65a12b8218a8c69"
sys.path.insert(0, str(ROOT / "scripts"))


def load(path: Path):
    return json.loads(path.read_text())


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)


def write_inventory(root: Path) -> None:
    """The repo's THIRD_PARTY.json into a scratch root, without the vendored entries (their trees are not copied;
    the Vendored class tests that relation on its own tree)."""
    doc = load(ROOT / "executor/THIRD_PARTY.json")
    doc["entries"] = [e for e in doc["entries"] if e["relation"] != "vendored"]
    (root / "executor").mkdir()
    (root / "executor/THIRD_PARTY.json").write_text(json.dumps(doc))


PROVENANCE = {
    "schema_version": "1", "image_digest": IMAGE_ID, "captured_utc": "2026-10-06T12:00:00Z",
    "toolchain": {"nvcc": "13.0.88", "cuda_runtime": "13.0.96", "gxx": "13.3.0", "cmake": "3.28.3",
                  "ninja": "1.11.1", "glibc": "2.39"},
    "oracle_packages": {"torch": "2.13.0+cu130", "sglang": "0.5.19.dev3+gd4689328e", "flashinfer_python": "0.6.17",
                        "safetensors": "0.8.0", "triton": "3.7.1", "transformers": "5.12.1", "tokenizers": "0.22.2"},
}


class Inventory(unittest.TestCase):
    def test_repo_inventory_passes(self):
        cp = run()
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

    def test_inventory_matches_schema_and_spec_entries(self):
        doc = load(ROOT / "executor/THIRD_PARTY.json")
        Draft202012Validator(load(ROOT / "schemas/third_party.schema.json")).validate(doc)
        entries = {e["name"]: e for e in doc["entries"]}
        for name, archive in (("cudart_static", "libcudart_static.a"), ("cublasLt_static", "libcublasLt_static.a"),
                              ("culibos", "libculibos.a")):
            e = entries[name]
            self.assertEqual((e["relation"], e["revision"], e["files"]), ("linked", IMAGE_ID, [archive]))
            self.assertEqual(e["license"], "NVIDIA CUDA Toolkit EULA")
        self.assertEqual(entries["sglang"]["revision"], "d4689328ed1de81d5ce491f52b93f6edb77e0ffe")
        self.assertEqual((entries["sglang"]["license"], entries["sglang"]["relation"]), ("Apache-2.0", "math_reference"))
        fla = entries["flash-linear-attention"]
        self.assertEqual((fla["license"], fla["relation"]), ("MIT", "math_reference"))
        self.assertEqual({s["file"] for s in doc["system_libraries"]},
                         {"libcuda.so.1", "libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6",
                          "ld-linux-x86-64.so.2"})

    def test_p3b_vendored_entries(self):
        """M2.3 P3b (N2): FlashInfer 0.6.17 and its bundled CUTLASS 4.5.0, pinned by the wheel's RECORD."""
        entries = {e["name"]: e for e in load(ROOT / "executor/THIRD_PARTY.json")["entries"]}
        fi, cl = entries["flashinfer-vendored"], entries["cutlass-vendored"]
        for e in (fi, cl):
            self.assertEqual((e["relation"], e["root"], e["pins"]),
                             ("vendored", "executor/third_party/flashinfer-0.6.17", "RECORD"))
        self.assertEqual((fi["version"], fi["license"]), ("0.6.17", "Apache-2.0"))
        self.assertEqual(fi["license_files"], ["flashinfer_python-0.6.17.dist-info/licenses/LICENSE"])
        self.assertTrue(cl["version"].startswith("4.5.0 "))
        self.assertEqual(cl["license"], "BSD-3-Clause")
        self.assertTrue(all(f.startswith("flashinfer/data/cutlass/") for f in cl["files"]))
        self.assertFalse(any(f.startswith("flashinfer/data/cutlass/") for f in fi["files"]))
        version = (ROOT / cl["root"] / "flashinfer/data/cutlass/include/cutlass/version.h").read_text()
        for part, value in (("MAJOR", 4), ("MINOR", 5), ("PATCH", 0)):
            self.assertIn(f"#define CUTLASS_{part} {value}\n", version)

    def check_entry_mutation(self, mutate, needle):
        doc = load(ROOT / "executor/THIRD_PARTY.json")
        mutate(doc["entries"][3])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "THIRD_PARTY.json"
            p.write_text(json.dumps(doc))
            cp = run("--third-party", str(p))
        self.assertEqual(cp.returncode, 1, cp.stdout + cp.stderr)
        self.assertIn(needle, cp.stdout + cp.stderr)

    def test_entry_without_license_fails(self):
        self.check_entry_mutation(lambda e: e.pop("license"), "license")

    def test_entry_without_revision_fails(self):
        self.check_entry_mutation(lambda e: e.pop("revision"), "revision")

    def test_empty_license_fails(self):
        self.check_entry_mutation(lambda e: e.update(license=""), "license")

    def test_unknown_relation_fails(self):
        self.check_entry_mutation(lambda e: e.update(relation="copied"), "relation")

    def test_duplicate_name_fails(self):
        doc = load(ROOT / "executor/THIRD_PARTY.json")
        doc["entries"].append(copy.deepcopy(doc["entries"][0]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "THIRD_PARTY.json"
            p.write_text(json.dumps(doc))
            cp = run("--third-party", str(p))
        self.assertEqual(cp.returncode, 1)
        self.assertIn("duplicate", cp.stdout + cp.stderr)


def record_line(rel: str, data: bytes) -> str:
    """One wheel RECORD line (PEP 376/427): path,sha256=<urlsafe base64, no padding>,size."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{rel},sha256={digest},{len(data)}"


class Vendored(unittest.TestCase):
    """M2.3 C8: the `vendored` relation (tasks/m23-spec.md C8, N2)."""

    ROOT_REL = "executor/third_party/pkg-1.0"
    FILES = {"pkg/LICENSE": b"Apache License\n", "pkg/a.h": b"#pragma once\n", "pkg/b.cu": b"int b;\n"}

    def entry(self) -> dict:
        return {"name": "pkg", "version": "1.0", "source": "pkg-1.0 wheel", "revision": "sha256:" + "a" * 64,
                "license": "Apache-2.0", "files": sorted(self.FILES), "relation": "vendored",
                "root": self.ROOT_REL, "pins": "RECORD", "license_files": ["pkg/LICENSE"]}

    def tree(self, mutate_entry=None, mutate_tree=None) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = load(ROOT / "executor/THIRD_PARTY.json")
            doc["entries"] = [e for e in doc["entries"] if e["relation"] != "vendored"]
            entry = self.entry()
            if mutate_entry:
                mutate_entry(entry)
            doc["entries"].append(entry)
            (root / "executor").mkdir()
            (root / "executor/THIRD_PARTY.json").write_text(json.dumps(doc))
            vroot = root / self.ROOT_REL
            for rel, data in self.FILES.items():
                (vroot / rel).parent.mkdir(parents=True, exist_ok=True)
                (vroot / rel).write_bytes(data)
            (vroot / "RECORD").write_text("".join(record_line(r, b) + "\n" for r, b in sorted(self.FILES.items())))
            if mutate_tree:
                mutate_tree(root, vroot)
            return run("--root", str(root))

    def assert_fails(self, cp, needle):
        self.assertEqual(cp.returncode, 1, cp.stdout + cp.stderr)
        self.assertIn(needle, cp.stdout)

    def test_vendored_entry_passes(self):
        cp = self.tree()
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

    def test_schema_accepts_vendored(self):
        doc = load(ROOT / "executor/THIRD_PARTY.json")
        doc["entries"].append(self.entry())
        Draft202012Validator(load(ROOT / "schemas/third_party.schema.json")).validate(doc)

    def test_missing_licence_file_fails(self):
        self.assert_fails(self.tree(mutate_tree=lambda r, v: (v / "pkg/LICENSE").unlink()), "pkg/LICENSE")

    def test_licence_files_required(self):
        self.assert_fails(self.tree(lambda e: e.pop("license_files")), "license_files")
        self.assert_fails(self.tree(lambda e: e.update(license_files=[])), "license_files")

    def test_licence_file_must_be_listed(self):
        self.assert_fails(self.tree(lambda e: e.update(license_files=["pkg/NOTICE"])), "pkg/NOTICE")

    def test_files_list_required(self):
        self.assert_fails(self.tree(lambda e: e.update(files=[])), "files")

    def test_unlisted_vendored_file_fails(self):
        cp = self.tree(mutate_tree=lambda r, v: (v / "pkg/extra.h").write_text("int x;\n"))
        self.assert_fails(cp, "pkg/extra.h")

    def test_file_outside_any_root_fails(self):
        def add(root, _v):
            p = root / "executor/third_party/other/x.h"
            p.parent.mkdir(parents=True)
            p.write_text("int x;\n")
        self.assert_fails(self.tree(mutate_tree=add), "executor/third_party/other/x.h")

    def test_listed_file_missing_fails(self):
        self.assert_fails(self.tree(mutate_tree=lambda r, v: (v / "pkg/a.h").unlink()), "pkg/a.h")

    def test_modified_file_fails_its_pin(self):
        self.assert_fails(self.tree(mutate_tree=lambda r, v: (v / "pkg/b.cu").write_bytes(b"int c;\n")), "pkg/b.cu")

    def test_unpinned_file_fails(self):
        def drop(_r, v):
            lines = (v / "RECORD").read_text().splitlines()
            (v / "RECORD").write_text("".join(l + "\n" for l in lines if not l.startswith("pkg/a.h,")))
        self.assert_fails(self.tree(mutate_tree=drop), "pkg/a.h")

    def test_pin_without_file_fails(self):
        def extra(_r, v):
            with (v / "RECORD").open("a") as f:
                f.write(record_line("pkg/gone.h", b"x") + "\n")
        self.assert_fails(self.tree(mutate_tree=extra), "pkg/gone.h")

    def test_malformed_pin_line_fails(self):
        def bad(_r, v):
            with (v / "RECORD").open("a") as f:
                f.write("pkg/a.h,md5=x,1\n")
        self.assert_fails(self.tree(mutate_tree=bad), "RECORD")

    def test_root_outside_third_party_fails(self):
        self.assert_fails(self.tree(lambda e: e.update(root="executor/cuda/src")), "root")

    def test_vendored_fields_only_on_vendored(self):
        doc = load(ROOT / "executor/THIRD_PARTY.json")
        doc["entries"][0]["root"] = self.ROOT_REL
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "THIRD_PARTY.json"
            p.write_text(json.dumps(doc))
            cp = run("--third-party", str(p))
        self.assert_fails(cp, "root")

    def test_pins_required(self):
        self.assert_fails(self.tree(lambda e: e.pop("pins")), "pins")


class BuildJson(unittest.TestCase):
    def test_fixtures_match_the_schema(self):
        schema = load(ROOT / "schemas/native_build.schema.json")
        for name in ("build_ok.json", "build_extra_lib.json"):
            Draft202012Validator(schema).validate(load(FIXTURES / name))

    def test_mutant_is_an_optional_sdar_mutant_name(self):
        """SelfTune ST4 S4a: `mutant` (a `native_dev.py build --sdar-mutant NAME` build.json) is optional and one of
        the sdar mutants; any other name, or another type, fails both validators."""
        schema = load(ROOT / "schemas/native_build.schema.json")
        build = load(FIXTURES / "build_ok.json")
        self.assertNotIn("mutant", build)
        import check_third_party
        for name in ("causal", "fp32tie", "changing_k_short"):
            Draft202012Validator(schema).validate(dict(build, mutant=name))
            self.assertEqual(check_third_party.schema_errors("native_build", dict(build, mutant=name)), [])
        for bad in ("window", "", "CAUSAL", 1, None):
            with self.subTest(bad=bad):
                self.assertFalse(Draft202012Validator(schema).is_valid(dict(build, mutant=bad)))
                self.assertTrue(check_third_party.schema_errors("native_build", dict(build, mutant=bad)))

    def test_build_ok_passes(self):
        cp = run("--build-json", str(FIXTURES / "build_ok.json"))
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

    def test_build_extra_lib_fails_naming_both(self):
        cp = run("--build-json", str(FIXTURES / "build_extra_lib.json"))
        self.assertEqual(cp.returncode, 1)
        self.assertIn("libcublasLt.so.13", cp.stdout)
        self.assertIn("libnccl_static.a", cp.stdout)

    def mutated(self, mutate) -> subprocess.CompletedProcess:
        doc = load(FIXTURES / "build_ok.json")
        mutate(doc)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "build.json"
            p.write_text(json.dumps(doc))
            return run("--build-json", str(p))

    def test_extra_needed_alone_fails(self):
        cp = self.mutated(lambda d: d["outputs"][1]["needed"].append("libnvrtc.so.13"))
        self.assertEqual(cp.returncode, 1)
        self.assertIn("libnvrtc.so.13", cp.stdout)

    def test_extra_static_alone_fails(self):
        cp = self.mutated(lambda d: d["static_libs"].append("libcutlass.a"))
        self.assertEqual(cp.returncode, 1)
        self.assertIn("libcutlass.a", cp.stdout)

    def test_malformed_build_json_fails(self):
        cp = self.mutated(lambda d: d.pop("static_libs"))
        self.assertEqual(cp.returncode, 1)
        self.assertIn("static_libs", cp.stdout)
        cp = self.mutated(lambda d: d.update(image_digest="sha256:" + "0" * 63))
        self.assertEqual(cp.returncode, 1)
        self.assertIn("image_digest", cp.stdout)


class SourceHeaders(unittest.TestCase):
    def tree(self, source: str, name: str = "src/kernels/gdn.cu") -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_inventory(root)
            f = root / "executor/cuda" / name
            f.parent.mkdir(parents=True)
            f.write_text(source)
            return run("--root", str(root))

    def test_listed_math_reference_passes(self):
        cp = self.tree("// Gated delta rule.\n// math_reference: flash-linear-attention\n// math_reference: sglang\n"
                       "#include <cstdint>\n")
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

    def test_unlisted_math_reference_fails(self):
        cp = self.tree("/* Attention.\n * math_reference: flashattention\n */\n#include <cstdint>\n",
                       "src/kernels/attn.cuh")
        self.assertEqual(cp.returncode, 1)
        self.assertIn("flashattention", cp.stdout)

    def test_linked_entry_is_not_a_math_reference(self):
        cp = self.tree("// math_reference: cudart_static\nint x;\n", "src/host/x.cpp")
        self.assertEqual(cp.returncode, 1)
        self.assertIn("cudart_static", cp.stdout)

    def test_empty_math_reference_fails(self):
        cp = self.tree("// math_reference:\nint x;\n")
        self.assertEqual(cp.returncode, 1)

    def test_repo_sources_pass(self):
        cp = run("--root", str(ROOT))
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)


class Provenance(unittest.TestCase):
    def check(self, doc) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_inventory(root)
            (root / "executor/PROVENANCE.json").write_text(json.dumps(doc))
            return run("--root", str(root))

    def test_schema_accepts_the_x1_shape(self):
        Draft202012Validator(load(ROOT / "schemas/provenance.schema.json")).validate(PROVENANCE)
        self.assertEqual(self.check(PROVENANCE).returncode, 0)

    def test_bad_provenance_fails(self):
        for mutate, needle in ((lambda d: d["toolchain"].pop("nvcc"), "nvcc"),
                               (lambda d: d.update(image_digest="latest"), "image_digest"),
                               (lambda d: d["oracle_packages"].update(torch=""), "torch"),
                               (lambda d: d.update(extra=1), "extra")):
            doc = copy.deepcopy(PROVENANCE)
            mutate(doc)
            cp = self.check(doc)
            with self.subTest(needle=needle):
                self.assertEqual(cp.returncode, 1)
                self.assertIn(needle, cp.stdout)


class SchemaSubset(unittest.TestCase):
    """The stdlib validator in check_third_party.py agrees with jsonschema on accept/reject."""

    def test_agreement(self):
        import check_third_party as ctp
        cases = [("native_build", load(FIXTURES / "build_ok.json")), ("provenance", PROVENANCE),
                 ("third_party", load(ROOT / "executor/THIRD_PARTY.json"))]
        bad = copy.deepcopy(load(FIXTURES / "build_ok.json"))
        bad["outputs"][0]["bytes"] = True
        cases.append(("native_build", bad))
        bad = copy.deepcopy(PROVENANCE)
        bad["captured_utc"] = "2026-10-06T12:00:00Z\n"
        cases.append(("provenance", bad))
        bad = copy.deepcopy(load(ROOT / "executor/THIRD_PARTY.json"))
        bad["entries"][0]["files"] = []
        cases.append(("third_party", bad))
        bad = copy.deepcopy(load(FIXTURES / "build_ok.json"))
        bad["static_libs"] = ["libculibos.a", "libculibos.a"]
        cases.append(("native_build", bad))
        for schema, doc in cases:
            with self.subTest(schema=schema):
                ref = sorted(e.message for e in Draft202012Validator(load(ROOT / f"schemas/{schema}.schema.json"))
                             .iter_errors(doc))
                ours = ctp.schema_errors(schema, doc)
                self.assertEqual(bool(ref), bool(ours), (ref, ours))


class Wiring(unittest.TestCase):
    def test_make_validate_runs_the_checker(self):
        line = next(l for l in (ROOT / "Makefile").read_text().splitlines() if "scripts/check_third_party.py" in l
                    and "--package" not in l)
        self.assertIn("scripts/check_third_party.py", line)


if __name__ == "__main__":
    unittest.main()
