"""B1: the committed tiny source model (Q14) is reproducible, covers every family kind and is not gitignored."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/fixtures'
SRC = FIX / 'model_src'
GENERATOR = ROOT / 'scripts/fixtures/make_model_fixture.py'
ITEMSIZE = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8, 'U8': 1, 'U16': 2,
            'U32': 4, 'U64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1, 'BOOL': 1}


def tree_files(base: Path) -> dict:
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in sorted(base.rglob('*')) if p.is_file()}


def headers() -> dict:
    """name -> (shard, header_len, metadata, tensors) via a plain header read (no tooling under test)."""
    out = {}
    for p in sorted(SRC.glob('*.safetensors')):
        raw = p.read_bytes()
        n = struct.unpack('<Q', raw[:8])[0]
        h = json.loads(raw[8:8 + n])
        out[p.name] = (n, h.pop('__metadata__', None), h)
    return out


def lock_digest(files) -> str:
    lines = ''.join(f'{hashlib.sha256((SRC / f).read_bytes()).hexdigest()}  {f}\n'
                    for f in sorted(files, key=lambda s: s.encode()))
    return hashlib.sha256(lines.encode()).hexdigest()


class GeneratorTests(unittest.TestCase):
    def test_generator_rewrites_committed_fixture_byte_identically(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, str(GENERATOR), '--out', d], check=True, capture_output=True)
            produced = tree_files(Path(d))
        # model_src.inventory.json is the B3 expected output of scripts/model_inventory.py, not a generator file.
        committed = {k: v for k, v in tree_files(FIX).items()
                     if (k.startswith('model_src/') or k.startswith('model_src.')) and k != 'model_src.inventory.json'}
        self.assertEqual(sorted(produced), sorted(committed))
        for name in produced:
            self.assertEqual(produced[name], committed[name], name)

    def test_check_mode_passes_on_committed_tree(self):
        subprocess.run([sys.executable, str(GENERATOR), '--check'], check=True, capture_output=True)


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = headers()
        cls.tensors = {name: (shard, n, t) for shard, (n, _, ts) in cls.h.items() for name, t in ts.items()}

    def names(self, pattern):
        import re
        return sorted(k for k in self.tensors if re.fullmatch(pattern, k))

    def test_three_shards_of_real_kinds(self):
        self.assertEqual(len(self.h), 3)
        kinds = sorted(name.split('-')[0] + '-' + name.split('-')[1] for name in self.h)
        self.assertEqual(kinds, ['layer-00000', 'model-bf16', 'model-plefp8'])

    def test_nvfp4_quads_with_scalar_scales(self):
        weights = self.names(r'.*\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight')
        self.assertGreaterEqual(len(weights), 3)
        for w in weights:
            prefix = w[:-len('.weight')]
            dt = {s: self.tensors[prefix + s][2] for s in ('.weight', '.weight_scale', '.weight_scale_2', '.input_scale')}
            self.assertEqual(dt['.weight']['dtype'], 'U8')
            self.assertEqual(dt['.weight_scale']['dtype'], 'F8_E4M3')
            self.assertEqual((dt['.weight_scale_2']['dtype'], dt['.weight_scale_2']['shape']), ('F32', []))
            self.assertEqual((dt['.input_scale']['dtype'], dt['.input_scale']['shape']), ('F32', []))
            o, half = dt['.weight']['shape']
            self.assertEqual(dt['.weight_scale']['shape'], [o, half * 2 // 16])

    def test_expert_metadata_ranges_match_names(self):
        import re
        shard = [s for s in self.h if s.startswith('layer-')][0]
        _, md, ts = self.h[shard]
        self.assertEqual(sorted(md), ['expert_end', 'expert_start', 'format', 'layer'])
        experts = {int(re.search(r'experts\.(\d+)\.', k).group(1)) for k in ts}
        self.assertEqual(experts, set(range(int(md['expert_start']), int(md['expert_end']))))

    def test_bf16_one_byte_misaligned_and_other_shards_aligned(self):
        mis = []
        for name, (shard, n, t) in self.tensors.items():
            if (8 + n + t['data_offsets'][0]) % ITEMSIZE[t['dtype']]:
                mis.append((shard, t['dtype'], (8 + n + t['data_offsets'][0]) % 2))
        self.assertIn(('model-bf16-00001.safetensors', 'BF16', 1), mis)
        self.assertTrue(all(s.startswith('model-bf16-') for s, _, _ in mis))

    def test_ple_table_module_scale_index_mtp_vision(self):
        self.assertEqual(len(self.names(r'.*\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight')), 4)
        module = self.names(r'.*\.ple\.(key_proj|value_proj|conv1d|norm_conv|norm_key|norm_query)\.weight')
        self.assertEqual(len(module), 6)
        self.assertTrue(all(self.tensors[m][2]['dtype'] == 'BF16' for m in module))
        self.assertEqual(self.tensors[self.names(r'.*ngram_embedding\.weight_scale')[0]][2]['shape'], [1])
        index = self.names(r'.*ple_embedding\.(layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes)')
        self.assertEqual([self.tensors[i][2]['dtype'] for i in index], ['I64'] * 3)
        self.assertTrue(self.names(r'mtp\..*') and self.names(r'model\.visual\..*'))
        big = max(t['data_offsets'][1] - t['data_offsets'][0] for _, _, t in self.tensors.values())
        self.assertGreater(big, 2 * 65536, 'a tensor must span >= 3 64 KiB staging buffers (R5)')

    def test_index_and_payload_stubs(self):
        index = json.loads((SRC / 'model.safetensors.index.json').read_bytes())
        self.assertEqual(index['weight_map'], {k: v[0] for k, v in self.tensors.items()})
        self.assertEqual(index['metadata']['total_size'], sum((SRC / s).stat().st_size for s in self.h))
        for f in ('config.json', 'hf_quant_config.json', 'generation_config.json', 'preprocessor_config.json',
                  'video_preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                  'merges.txt', 'chat_template.jinja'):
            self.assertTrue((SRC / f).is_file(), f)

    def test_token_map_is_stored_zip_with_data_descriptor(self):
        with zipfile.ZipFile(SRC / 'token_map.pt') as z:
            infos = z.infolist()
            self.assertEqual(infos[0].filename.split('/')[-1], 'data.pkl')
            self.assertEqual(infos[0].header_offset, 0)
            for i in infos:
                self.assertEqual(i.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(i.flag_bits, 0x808)
            import pickletools
            ops = {op.name for op, _, _ in pickletools.genops(z.read(infos[0]))}
        self.assertTrue({'BININT1', 'BININT2', 'BININT', 'MARK', 'APPENDS', 'BINPUT'} <= ops)
        self.assertFalse(ops & {'GLOBAL', 'STACK_GLOBAL', 'REDUCE', 'FRAME'})

    def test_hf_tree_has_one_non_lfs_config_size_difference(self):
        trees = list((SRC / '.cache/huggingface/trees').glob('*.json'))
        self.assertEqual(len(trees), 1)
        files = json.loads(trees[0].read_bytes())['files']
        diffs = [k for k, v in files.items() if (SRC / k).is_file() and v['size'] != (SRC / k).stat().st_size]
        self.assertEqual(diffs, ['config.json'])
        self.assertNotIn('lfs_sha256', files['config.json'])
        for shard in self.h:
            self.assertEqual(files[shard]['lfs_sha256'], hashlib.sha256((SRC / shard).read_bytes()).hexdigest())

    def test_fixture_lock_uses_lock_definitions(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from cere_contracts import load_json, validate
        lock = load_json(FIX / 'model_src.lock.json')
        validate('deployment_lock', lock)
        self.assertEqual(lock['model_content_sha256'], lock_digest(
            [*self.h, 'model.safetensors.index.json', 'config.json', 'hf_quant_config.json']))
        self.assertEqual(lock['tokenizer_sha256'],
                         lock_digest(['merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']))
        self.assertEqual(lock['template_sha256'], hashlib.sha256((SRC / 'chat_template.jinja').read_bytes()).hexdigest())
        top = [p.name for p in SRC.iterdir() if p.is_file()]
        self.assertEqual(lock['artifact_sha256'], lock_digest(top))


class GitignoreTests(unittest.TestCase):
    def test_fixture_and_example_payload_not_ignored(self):
        paths = [p.relative_to(ROOT).as_posix() for p in SRC.rglob('*') if p.is_file()]
        manifest = json.loads((ROOT / 'examples/artifact/manifest.json').read_bytes())
        paths += ['examples/artifact/' + f['path'] for f in manifest['files']]
        paths += ['examples/artifact/model-bf16-00001.safetensors']
        r = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], cwd=ROOT, input='\n'.join(paths),
                           capture_output=True, text=True)
        self.assertEqual(r.stdout, '', 'ignored: ' + r.stdout)
        self.assertEqual(r.returncode, 1)


if __name__ == '__main__':
    unittest.main()
