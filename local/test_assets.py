"""Small tests for restoration integrity and write boundaries."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('sanw_assets', Path(__file__).with_name('assets.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'local/assets/chunks').mkdir(parents=True)
        self.payload = b'fixed training feature bytes'
        self.member = {'path': 'results/train/features.npz', **identity(self.payload), 'restore_group': 'train-dev'}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            z.writestr(self.member['path'], self.payload)
            z.writestr('results/test/features.npz', b'unopened benchmark')
        data = buf.getvalue()
        chunk = {'path': 'local/assets/chunks/one.chunk', **identity(data)}
        (self.root / chunk['path']).write_bytes(data)
        self.archive = {'name': 'fixture.zip', 'format': 'zip', **identity(data),
                        'assembled_path': 'local/assets/archives/fixture.zip', 'chunks': [chunk],
                        'members': [self.member, {'path': 'results/test/features.npz',
                         **identity(b'unopened benchmark'), 'restore_group': 'archive-only'}]}
        self.manifest = {'schema': 'sanw-local-assets-v1', 'archives': [self.archive], 'raw_assets': []}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.root / 'local/assets_manifest.json').write_text(json.dumps(self.manifest))

    def assets(self):
        return module.Assets(self.root)

    def test_dry_run_writes_nothing(self):
        self.assets().restore('train-dev')
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / self.archive['assembled_path']).exists())

    def test_restore_is_exact_idempotent_and_excludes_benchmark(self):
        self.assets().restore('train-dev', True)
        target = self.root / self.member['path']
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse((self.root / 'results/test/features.npz').exists())
        self.assertEqual(self.assets().restore('train-dev', True)['restores'], [])

    def test_changed_destination_stops_before_assembly(self):
        target = self.root / self.member['path']
        target.parent.mkdir(parents=True)
        target.write_bytes(b'preserve this result')
        with self.assertRaises(ValueError):
            self.assets().restore('train-dev', True)
        self.assertEqual(target.read_bytes(), b'preserve this result')
        self.assertFalse((self.root / self.archive['assembled_path']).exists())

    def test_changed_chunk_cannot_publish_archive(self):
        (self.root / self.archive['chunks'][0]['path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            self.assets().assemble(execute=True)
        self.assertFalse((self.root / self.archive['assembled_path']).exists())

    def test_traversal_and_symlink_rejected(self):
        with self.assertRaises(ValueError):
            module.safe_path(self.root, '../outside')
        (self.root / 'link').symlink_to(self.root / 'local', target_is_directory=True)
        with self.assertRaises(ValueError):
            module.safe_path(self.root, 'link/target')

    def test_member_hash_failure_never_publishes_member(self):
        self.member['sha256'] = '0' * 64
        self.save()
        with self.assertRaises(ValueError):
            self.assets().restore('train-dev', True)
        self.assertFalse((self.root / self.member['path']).exists())

    def test_direct_fit_tar_restores_only_selected_run(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:xz') as tar:
            info = tarfile.TarInfo('results/run/selected.npz')
            info.size = len(self.payload)
            tar.addfile(info, io.BytesIO(self.payload))
            info = tarfile.TarInfo('src/frozen.py')
            info.size = 3
            tar.addfile(info, io.BytesIO(b'old'))
        data = buf.getvalue()
        path = 'local/assets/fit.tar.xz'
        (self.root / path).write_bytes(data)
        self.manifest['archives'] = [{'name': 'fit.tar.xz', 'path': path, 'format': 'tar.xz', **identity(data),
                                     'members': [{'path': 'results/run/selected.npz', **identity(self.payload),
                                                  'restore_group': 'fits'}]}]
        self.save()
        self.assets().restore('fits', True)
        self.assertEqual((self.root / 'results/run/selected.npz').read_bytes(), self.payload)
        self.assertFalse((self.root / 'src/frozen.py').exists())


if __name__ == '__main__':
    unittest.main()
