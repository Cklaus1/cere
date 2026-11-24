"""ABI v3 migration (spec H1): nothing outside tasks/ still names the v2 loader or the v2 ABI, except the migration
note in executor/README.md and CHANGELOG.md's history (with its pending fragments, changes/). ABI v4 (PerfInt lever
registry): the README carries the v4 migration note (cere_levers, the two lever functions, the removed build options).
ABI v5 (SelfTune ST2): the header is v5 and the README carries the v5 note (the slot-selection block, the renamed
functions, the desc's family and calibrate_windows)."""
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Built from parts so this file does not match itself.
STALE = [("cere_model" + "_load").encode(), ("ABI " + "v2").encode(), ("ABI version " + "2").encode()]
SKIPPED_DIRS = {".git", ".venv", "target", "__pycache__", ".pytest_cache", "build", "tasks"}
# The long-context DICE-8B prompt sets (scripts/tune/dice_longctx.py) quote repo files verbatim at a pinned historical
# commit, digest-committed, so they keep that commit's names; they are prompt text, not code or docs.
SKIPPED_PREFIXES = ("bench/results/", "traces/private/", "bench/prompts/dice_longctx_")
MIGRATION_NOTE = "executor/README.md"
NOTE_HEADING = "## ABI v3 migration"
V4_NOTE_HEADING = "## ABI v4 migration"
V5_NOTE_HEADING = "## ABI v5 migration"


def repo_files():
    # os.walk, pruning the skipped dirs before it descends: other test processes create and delete temp dirs under
    # build/ and bench/results/ while this runs (make test-python runs modules in parallel), and Python 3.11's rglob
    # raises FileNotFoundError on a directory removed mid-walk.
    paths = []
    for top, dirs, files in os.walk(ROOT):
        rel_top = Path(top).relative_to(ROOT)
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS
                   and not f"{(rel_top / d).as_posix()}/".startswith(SKIPPED_PREFIXES)]
        paths += [Path(top, f) for f in files]
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & SKIPPED_DIRS or rel.as_posix().startswith(SKIPPED_PREFIXES) or not path.is_file():
            continue
        yield rel.as_posix(), path


def migration_note(text: str, heading: str = NOTE_HEADING) -> str:
    start = text.index(heading)
    nxt = re.search(r"^## ", text[start + len(heading):], re.M)
    return text[start:start + len(heading) + nxt.start()] if nxt else text[start:]


class AbiMigrationTest(unittest.TestCase):
    def test_no_stale_v2_names_outside_note_and_changelog(self):
        offenders = []
        for rel, path in repo_files():
            if rel == "CHANGELOG.md" or rel.startswith("changes/"):
                continue
            data = path.read_bytes()
            if rel == MIGRATION_NOTE:
                text = data.decode()
                note = migration_note(text)
                data = text.replace(note, "").encode()
            offenders += [f"{rel}: {needle.decode()}" for needle in STALE if needle in data]
        self.assertEqual(offenders, [])

    def test_migration_note_describes_v3(self):
        note = migration_note((ROOT / MIGRATION_NOTE).read_text())
        for name in ("cere_model_create", "cere_model_put", "cere_model_seal", "cere_model" + "_load",
                     "operations", "finish", "out_tokens", "stop_token_ids", "cere_executor_abi_version"):
            self.assertIn(name, note)

    def test_migration_note_describes_v4(self):
        note = migration_note((ROOT / MIGRATION_NOTE).read_text(), V4_NOTE_HEADING)
        for name in ("cere_levers", "levers", "120 to 128", "cere_model_levers", "cere_model_set_levers",
                     "CERE_ERR_BUSY", "CERE_LEVER_DECODE_GRAPH", "CERE_LEVER_MOE_BUILD", "draft_tokens",
                     "tests/fixtures/executor_abi_v4_layout.json"):
            self.assertIn(name, note)

    def test_migration_note_describes_v5(self):
        note = migration_note((ROOT / MIGRATION_NOTE).read_text(), V5_NOTE_HEADING)
        for name in ("cere_slot_selection", "cere_slot_choice", "selection", "128 to 136", "family",
                     "calibrate_windows", "cere_model_selection", "cere_model_set_selection", "CERE_ERR_BUSY",
                     "CERE_LEVER_DRAFT_WINDOW", "block_version", "tests/fixtures/executor_abi_v5_layout.json"):
            self.assertIn(name, note)

    def test_header_is_v5_without_loader(self):
        header = (ROOT / "executor/include/cere_executor.h").read_text()
        self.assertIn("#define CERE_EXECUTOR_ABI_VERSION UINT32_C(5)", header)
        self.assertIn("cere_status cere_model_set_selection(cere_executor *executor, cere_model *model, "
                      "const cere_slot_selection *selection);", header)
        self.assertNotIn("} cere_levers;", header)
        self.assertNotIn("Design contract only", header)
        self.assertIn("descriptors and input token IDs are copied before return; output buffers are leased until the "
                      "terminal poll", header)


if __name__ == "__main__":
    unittest.main()
