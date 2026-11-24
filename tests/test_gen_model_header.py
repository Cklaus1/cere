"""tasks/selftune-spec.md D1 / ST1a: executor/tools/gen_model_header.py (the constants header of a model descriptor)
and the precheck that keeps the generated header from drifting (cere_contracts.model_header_check, run by
validate_repo.py)."""
from pathlib import Path
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cere_contracts import (BUILD_DESCRIPTOR, FAMILY_MACROS, ContractError, executor_constant,  # noqa: E402
                            gen_model_header, generated_constants, load_json, model_header_check)

GEN = gen_model_header(ROOT)
QWEN = ROOT / "executor/models/qwen38-flash-next.json"
DICE = ROOT / "executor/models/dice-8b.json"
INKLING = ROOT / "executor/models/inkling-2-mini.json"


class Generator(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def run_main(self, src: Path, dst: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(ROOT / "executor/tools/gen_model_header.py"), str(src), str(dst)],
                              capture_output=True, text=True, cwd=self.tmp)

    def test_both_descriptors_render_byte_deterministically(self):
        for path in (QWEN, DICE):
            with self.subTest(path=path.name):
                a, b = self.tmp / "a" / "h.h", self.tmp / "b.h"
                self.assertEqual(self.run_main(path, a).returncode, 0)
                self.assertEqual(self.run_main(path, b).returncode, 0)
                self.assertEqual(a.read_bytes(), b.read_bytes())
                text = a.read_text(encoding="ascii")
                self.assertEqual(text, GEN.render(load_json(path)))
                self.assertNotIn(str(ROOT), text)
                self.assertNotIn(str(self.tmp), text)
                names = [line.split(" = ")[0].split()[-1].split("[")[0] for line in text.splitlines()
                         if line.startswith("inline constexpr")]
                self.assertEqual(names, sorted(names))

    def test_qwen_header_values(self):
        text = GEN.render(load_json(QWEN))
        for decl in ("inline constexpr uint32_t kNumHiddenLayers = 48;",
                     "inline constexpr uint32_t kHiddenSize = 2560;",
                     "inline constexpr uint32_t kNumExperts = 512;",
                     "inline constexpr uint32_t kVocabSize = 248320;",
                     "inline constexpr uint32_t kMropeSection[3] = {11, 11, 10};",
                     "inline constexpr uint32_t kPleLayerIds[1] = {2};",
                     "inline constexpr bool kHasDecode = false;",
                     "inline constexpr bool kHasVision = true;"):
            self.assertIn(decl, text)
        self.assertEqual(generated_constants(ROOT)["kHiddenSize"], 2560)

    def test_dice_header_carries_the_decode_policy(self):
        text = GEN.render(load_json(DICE))
        for decl in ('inline constexpr char kModelId[] = "dice-8b";',
                     "inline constexpr bool kHasDecode = true;",
                     "inline constexpr bool kHasVision = false;",
                     'inline constexpr char kDecodePolicy[] = "block_diffusion";',
                     "inline constexpr uint32_t kDecodeBlockSize = 4;",
                     "inline constexpr uint32_t kNumHiddenLayers = 36;"):
            self.assertIn(decl, text)

    @unittest.skipUnless(shutil.which("c++"), "no host C++ compiler")
    def test_headers_compile(self):
        for path in (QWEN, DICE):
            with self.subTest(path=path.name):
                header = self.tmp / f"{path.stem}.h"
                header.write_text(GEN.render(load_json(path)), encoding="ascii")
                unit = self.tmp / f"{path.stem}.cpp"
                unit.write_text(f'#include "{header.name}"\nstatic_assert(cere::gen::kNumHiddenLayers > 0);\n')
                done = subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                                       str(unit)], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)

    def test_refusals(self):
        base = load_json(QWEN)
        cases = (
            (lambda d: d.pop("model_id"), "no model_id"),
            (lambda d: d["dimensions"]["mrope_section"].update(value=[[11], [11]]), "dimensions.mrope_section"),
            (lambda d: d["dimensions"]["hidden_size"].update(value=float("inf")), "dimensions.hidden_size"),
            (lambda d: d["dimensions"].update(stray=3), "dimensions.stray: plain value"),
            (lambda d: d["layers"].update(kinds=[]), "layers.kinds"),
        )
        for mutate, needle in cases:
            with self.subTest(needle=needle):
                data = copy.deepcopy(base)
                mutate(data)
                with self.assertRaises(GEN.DescriptorError) as cm:
                    GEN.render(data)
                self.assertIn(needle, str(cm.exception))

    def test_main_refuses_a_misnamed_descriptor(self):
        src = self.tmp / "other.json"
        src.write_bytes(QWEN.read_bytes())
        done = self.run_main(src, self.tmp / "h.h")
        self.assertEqual(done.returncode, 1)
        self.assertIn("file name is not <model_id>.json", done.stderr)
        self.assertFalse((self.tmp / "h.h").exists())

    def test_family_follows_model_type(self):
        """SelfTune ST4 S3a: kFamily and CERE_GEN_FAMILY are the CERE_FAMILY_* value of `model_type` (the two
        tables agree), and a model_type with no family is refused."""
        header = (ROOT / "executor/include/cere_executor.h").read_text()
        for model_type, value in GEN.FAMILIES.items():
            with self.subTest(model_type=model_type):
                macro = FAMILY_MACROS.get(model_type, model_type.upper())
                self.assertIn(f"#define CERE_FAMILY_{macro} UINT32_C({value})", header)
        for path, value in ((QWEN, 0), (DICE, 1), (INKLING, 3)):
            with self.subTest(path=path.name):
                text = GEN.render(load_json(path))
                self.assertIn(f"inline constexpr uint32_t kFamily = {value};", text)
                self.assertIn(f"#define CERE_GEN_FAMILY {value}\n", text)
        data = load_json(DICE)
        data["model_type"]["value"] = "llama"
        with self.assertRaises(GEN.DescriptorError) as cm:
            GEN.render(data)
        self.assertIn("model_type: 'llama' is no family", str(cm.exception))

    def test_hip_build_refuses_another_family_by_name(self):
        """ST4 S3a: executor/hip/build.sh builds the Qwen family only; CERE_MODEL=dice-8b (family sdar) and a model
        with no descriptor are refused (exit 2) before anything under /src is read or built."""
        for model, needle in (("dice-8b", "family sdar is not built by the HIP port"),
                              ("no-such-model", "names no descriptor under executor/models/"),
                              ("../x", "names no descriptor")):
            with self.subTest(model=model):
                done = subprocess.run(["bash", str(ROOT / "executor/hip/build.sh")], capture_output=True, text=True,
                                      cwd=self.tmp, env=dict(os.environ, CERE_MODEL=model), timeout=60)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn(needle, done.stderr)
                self.assertEqual(list(self.tmp.iterdir()), [])


class HeaderCheck(unittest.TestCase):
    """model_header_check on a copy of the files it reads."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for rel in ("executor/cuda/src", "executor/tools", "executor/models"):
            shutil.copytree(ROOT / rel, self.root / rel)
        for rel in ("executor/CMakeLists.txt", "Makefile"):
            shutil.copy(ROOT / rel, self.root / rel)

    def refused(self, needle):
        with self.assertRaises(ContractError) as cm:
            model_header_check(self.root)
        self.assertIn(needle, str(cm.exception))

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text()
        self.assertEqual(text.count(old), 1, old)
        path.write_text(text.replace(old, new))

    def test_tree_passes(self):
        self.assertGreater(model_header_check(ROOT), 50)
        self.assertEqual(model_header_check(self.root), model_header_check(ROOT))

    def test_unknown_generated_constant(self):
        self.edit("executor/cuda/src/model/model.h", "gen::kNumExperts;", "gen::kNumExpert;")
        self.refused("gen::kNumExpert is not a constant of")

    def test_descriptor_rename_breaks_the_uses(self):
        data = load_json(self.root / BUILD_DESCRIPTOR)
        data["dimensions"]["num_routed_experts"] = data["dimensions"].pop("num_experts")
        (self.root / BUILD_DESCRIPTOR).write_text(json.dumps(data))
        self.refused("gen::kNumExperts is not a constant of")

    def test_committed_header_is_refused(self):
        (self.root / "executor/cuda/src/model/cere_model_constants.h").write_text("#pragma once\n")
        self.refused("generated by the build, never committed")

    def test_cmake_must_name_generator_and_descriptor(self):
        self.edit("executor/CMakeLists.txt",
                  "DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_model_header.py ",
                  "DEPENDS ")
        self.refused("executor/CMakeLists.txt: no add_custom_command")

    def test_cmake_default_model_must_be_the_build_descriptor(self):
        self.edit("executor/CMakeLists.txt", "set(CERE_MODEL qwen38-flash-next CACHE STRING ",
                  "set(CERE_MODEL dice-8b CACHE STRING ")
        self.refused("with CERE_MODEL defaulting to executor/models/qwen38-flash-next.json")

    def test_family_guarded_uses_are_checked_against_that_family(self):
        """ST4 S3a: a `gen::` use under `#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR` must be a constant of DICE-8B's
        header (kNumExperts is not); its `#else` branch is checked against every other family's (Inkling 2 Mini's
        header has no kIntermediateSize: its experts' width is kMoeIntermediateSize)."""
        unit = self.root / "executor/cuda/src/host/guarded.h"
        good = ("#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR\nx = gen::kIntermediateSize;\n#else\n"
                "x = gen::kNumExperts;\n#endif\n")
        unit.write_text(good)
        self.assertEqual(model_header_check(self.root), model_header_check(ROOT) + 2)
        unit.write_text("#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR\nx = gen::kNumExperts;\n#endif\n")
        self.refused("guarded.h: gen::kNumExperts is not a constant of executor/models/dice-8b.json's header")
        unit.write_text(good.replace("gen::kNumExperts", "gen::kIntermediateSize").replace(
            "x = gen::kIntermediateSize;\n#else", "x = gen::kHiddenSize;\n#else"))
        self.refused("guarded.h: gen::kIntermediateSize is not a constant of executor/models/inkling-2-mini.json's")

    def test_family_subdirectory_is_checked_against_that_family(self):
        """ST4 S3c: a file under a family's own subdirectory (`kernels/sdar/`) is compiled only into that family's
        library, so its `gen::` uses are checked against DICE-8B's header (kDecodeMaskTokenId is one, kNumExperts is
        not), and the same use outside such a directory against the Qwen one."""
        unit = self.root / "executor/cuda/src/kernels/sdar/probe.h"
        unit.write_text("x = gen::kDecodeMaskTokenId;\n")
        self.assertEqual(model_header_check(self.root), model_header_check(ROOT) + 1)
        unit.write_text("x = gen::kNumExperts;\n")
        self.refused("probe.h: gen::kNumExperts is not a constant of executor/models/dice-8b.json's header")
        unit.unlink()
        (self.root / "executor/cuda/src/kernels/probe.h").write_text("x = gen::kDecodeMaskTokenId;\n")
        self.refused("probe.h: gen::kDecodeMaskTokenId is not a constant of executor/models/qwen38-flash-next.json's")

    def test_inkling_subdirectory_and_guard_are_checked_against_inkling(self):
        """Inkling 2 Mini S0: family inkling2_loop's macro is CERE_FAMILY_INKLING2 and its own subdirectory is
        `inkling/` (cere_contracts FAMILY_MACROS / FAMILY_DIRS); uses there are checked against its header."""
        unit = self.root / "executor/cuda/src/kernels/inkling/probe.h"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text("x = gen::kLoopNumPasses;\n")
        self.assertEqual(model_header_check(self.root), model_header_check(ROOT) + 1)
        unit.write_text("x = gen::kIntermediateSize;\n")
        self.refused("probe.h: gen::kIntermediateSize is not a constant of executor/models/inkling-2-mini.json's")
        unit.unlink()
        guarded = self.root / "executor/cuda/src/host/guarded.h"
        guarded.write_text("#if CERE_GEN_FAMILY == CERE_FAMILY_INKLING2\nx = gen::kLoopNumPasses;\n#endif\n")
        self.assertEqual(model_header_check(self.root), model_header_check(ROOT) + 1)
        guarded.write_text("#if CERE_GEN_FAMILY == CERE_FAMILY_INKLING2\nx = gen::kDecodeMaskTokenId;\n#endif\n")
        self.refused("guarded.h: gen::kDecodeMaskTokenId is not a constant of executor/models/inkling-2-mini.json's")

    def test_executor_constant_reads_the_build_family_branch(self):
        """ST4 S3a: host/mrope.h defines kSpatialMergeSize once per family branch; the descriptor's `executor:`
        source reads the Qwen one (BUILD_DESCRIPTOR's family), and two definitions in one branch are refused."""
        source = "executor:executor/cuda/src/host/mrope.h#kSpatialMergeSize"
        self.assertEqual(executor_constant(self.root, source), 2)
        self.edit("executor/cuda/src/host/mrope.h", "#elif CERE_GEN_FAMILY == CERE_FAMILY_INKLING2\n",
                  "constexpr uint32_t kSpatialMergeSize = 3;\n#elif CERE_GEN_FAMILY == CERE_FAMILY_INKLING2\n")
        with self.assertRaises(ContractError) as cm:
            executor_constant(self.root, source)
        self.assertIn("2 constexpr definitions of kSpatialMergeSize", str(cm.exception))

    def test_makefile_must_name_generator_and_descriptor(self):
        self.edit("Makefile", "$(HOST_MODEL_HEADER): executor/tools/gen_model_header.py executor/models/",
                  "$(HOST_MODEL_HEADER): executor/tools/gen_model_header.py executor/")
        self.refused("Makefile: the $(HOST_MODEL_HEADER) rule")

    def test_unrenderable_descriptor_is_refused(self):
        data = load_json(self.root / "executor/models/dice-8b.json")
        data["dimensions"]["hidden_size"]["value"] = [[1]]
        (self.root / "executor/models/dice-8b.json").write_text(json.dumps(data))
        self.refused("dice-8b.json: dimensions.hidden_size")


if __name__ == "__main__":
    unittest.main()
