"""scripts/pin_configs.py: the one generated pin file `tests/fixtures/config_pins.json`, its `--check` (wired into
`make validate`) failing on any drift from the committed runtime configs until `--write` regenerates it."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pin_configs  # noqa: E402
from cere_contracts import canonical_digest  # noqa: E402

LIVE = 'configs/runtime/sglang-t4-qwen38.toml'
NATIVE = 'configs/runtime/native-g0.toml'


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = pin_configs.main(list(argv))
    return rc, out.getvalue()


class PinConfigs(unittest.TestCase):
    def setUp(self):
        """A root with the committed runtime configs and pin file."""
        self.root = Path(tempfile.mkdtemp(prefix='test_pin_configs_'))
        self.addCleanup(shutil.rmtree, self.root)
        shutil.copytree(ROOT / 'configs/runtime', self.root / 'configs/runtime')
        (self.root / 'tests/fixtures').mkdir(parents=True)
        shutil.copy(ROOT / pin_configs.PINS, self.root / pin_configs.PINS)

    def pins(self):
        return json.loads((self.root / pin_configs.PINS).read_text())

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text()
        self.assertEqual(text.count(old), 1, old)
        path.write_text(text.replace(old, new))

    def test_the_committed_pins_match_the_committed_configs(self):
        self.assertEqual(pin_configs.drift(ROOT), [])
        rc, out = run('--check')
        self.assertEqual(rc, 0, out)
        pins = pin_configs.load()
        self.assertEqual(pins['generated_by'], 'python3 scripts/pin_configs.py --write')
        self.assertEqual(sorted(pins['config_hash']),
                         sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'configs/runtime').glob('*.toml')))
        self.assertEqual(list(pins['masked_toml_sha256']), [NATIVE])
        self.assertEqual((ROOT / pin_configs.PINS).read_text(), pin_configs.render(pin_configs.compute()))

    def test_a_config_edit_fails_check_until_write(self):
        before = self.pins()
        self.edit(LIVE, 'queue_depth = 64', 'queue_depth = 65')
        now = canonical_digest(tomllib.loads((self.root / LIVE).read_text()))
        rc, out = run('--check', '--root', str(self.root))
        self.assertEqual(rc, 1, out)
        self.assertIn(f"DRIFT: config_hash {LIVE}: pinned {before['config_hash'][LIVE]}, now {now}", out)
        self.assertIn('run `python3 scripts/pin_configs.py --write` and commit the file', out)
        self.assertEqual(out.count('DRIFT:'), 1, out)
        self.assertEqual(self.pins(), before)  # --check writes nothing
        rc, out = run('--write', '--root', str(self.root))
        self.assertEqual((rc, out.splitlines()[0]), (0, f'pin_configs: wrote {pin_configs.PINS} (1 change(s))'))
        after = self.pins()
        self.assertEqual(after['config_hash'][LIVE], now)
        self.assertEqual({k: v for k, v in after['config_hash'].items() if k != LIVE},
                         {k: v for k, v in before['config_hash'].items() if k != LIVE})
        self.assertEqual(run('--check', '--root', str(self.root))[0], 0)
        self.assertEqual(run('--write', '--root', str(self.root))[1],
                         f'pin_configs: {pin_configs.PINS} up to date\n')

    def test_a_relocked_library_moves_the_config_hash_but_not_the_masked_pin(self):
        """`library_sha256` is a `masked` field: a relock moves native-g0's config_hash only."""
        lib = tomllib.loads((self.root / NATIVE).read_text())['deployments'][0]['native']['library_sha256']
        self.edit(NATIVE, lib, '0' * 64)
        drift = pin_configs.drift(self.root)
        self.assertEqual(len(drift), 1, drift)
        self.assertTrue(drift[0].startswith(f'config_hash {NATIVE}: pinned '), drift)

    def test_an_unmasked_native_g0_edit_moves_both_pins(self):
        self.edit(NATIVE, 'bind = "127.0.0.1:18309"', 'bind = "127.0.0.1:18399"')
        self.assertEqual([line.split(':')[0] for line in pin_configs.drift(self.root)],
                         [f'config_hash {NATIVE}', f'masked_toml_sha256 {NATIVE}'])

    def test_an_added_or_removed_config_fails_check(self):
        shutil.copy(self.root / LIVE, self.root / 'configs/runtime/zz-new.toml')
        (self.root / 'configs/runtime/dev-single-gpu.toml').unlink()
        drift = pin_configs.drift(self.root)
        self.assertEqual(drift, ['config_hash configs/runtime/dev-single-gpu.toml: pinned, but the config is gone',
                                 'config_hash configs/runtime/zz-new.toml: not pinned (now '
                                 f"{self.pins()['config_hash'][LIVE]})"])
        self.assertEqual(run('--write', '--root', str(self.root))[0], 0)
        self.assertEqual(pin_configs.drift(self.root), [])
        self.assertNotIn('configs/runtime/dev-single-gpu.toml', self.pins()['config_hash'])

    def test_a_hand_edited_or_missing_pin_file_fails_check(self):
        pins = self.pins()
        pins['config_hash'][LIVE] = 'f' * 64
        (self.root / pin_configs.PINS).write_text(pin_configs.render(pins))
        self.assertEqual(pin_configs.drift(self.root)[0].split(':')[0], f'config_hash {LIVE}')
        pins = pin_configs.compute(self.root)
        (self.root / pin_configs.PINS).write_text(json.dumps(pins))  # the right values, not `--write`'s form
        self.assertEqual(pin_configs.drift(self.root),
                         [f'{pin_configs.PINS} is not in the form `--write` writes (edited by hand?)'])
        (self.root / pin_configs.PINS).unlink()
        self.assertEqual(pin_configs.drift(self.root), [f'{pin_configs.PINS} is missing'])
        self.assertEqual(run('--check', '--root', str(self.root))[0], 1)
        self.assertEqual(run('--write', '--root', str(self.root))[0], 0)
        self.assertEqual(self.pins(), pins)

    def test_one_mode_is_required(self):
        for argv in ([], ['--check', '--write']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as caught:
                pin_configs.main(argv)
            self.assertEqual(caught.exception.code, 2)

    def test_make_validate_runs_the_check(self):
        line = next(line for line in (ROOT / 'Makefile').read_text().splitlines()
                    if 'scripts/check_third_party.py' in line and '--package' not in line)
        self.assertIn('python3 scripts/pin_configs.py --check', line)


if __name__ == '__main__':
    unittest.main()
