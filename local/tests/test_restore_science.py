"""Small scientific snapshot fixtures; no arrays, training, or outcome access."""
import hashlib
import importlib.util
import io
import tarfile
import tempfile
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    'restore_science', Path(__file__).resolve().parents[1] / 'restore_science.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class SnapshotTests(unittest.TestCase):
    def fixture(self, root, name='src/example.py', content=b'new scientific bytes\n', previous=None):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:xz') as tar:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        raw = archive.getvalue()
        chunk = root / 'local/assets/chunks/fixture.chunk'
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_bytes(raw)
        row = {'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        if previous is not None:
            row['previous_git_blob'] = hashlib.sha1(
                b'blob ' + str(len(previous)).encode() + b'\0' + previous).hexdigest()
        return {'files': [row], 'parts': [{'path': str(chunk.relative_to(root)),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}],
                'archive_bytes': len(raw), 'archive_sha256': hashlib.sha256(raw).hexdigest()}

    def test_empty_restore_and_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            self.assertFalse(m.restore(root, manifest, check=True)['verified'])
            self.assertEqual(m.restore(root, manifest)['restored_files'], 1)
            self.assertTrue(m.restore(root, manifest, check=True)['verified'])

    def test_recognized_base_blob_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = b'old published bytes\n'
            manifest = self.fixture(root, previous=old)
            path = root / 'src/example.py'
            path.parent.mkdir()
            path.write_bytes(old)
            m.restore(root, manifest)
            self.assertEqual(path.read_bytes(), b'new scientific bytes\n')

    def test_unknown_edit_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, previous=b'old published bytes\n')
            path = root / 'src/example.py'
            path.parent.mkdir()
            path.write_bytes(b'user changes\n')
            with self.assertRaises(ValueError):
                m.restore(root, manifest)
            self.assertEqual(path.read_bytes(), b'user changes\n')
            self.assertFalse((root / 'local/assets/science_snapshot').exists())

    def test_traversal_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root, name='../outside')
            with self.assertRaises(ValueError):
                m.restore(root, manifest)

    def test_corrupt_chunk_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            (root / manifest['parts'][0]['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                m.restore(root, manifest)
            self.assertFalse((root / 'src/example.py').exists())

    def test_member_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest['files'][0]['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                m.restore(root, manifest)
            self.assertFalse((root / 'src/example.py').exists())

    def test_dangling_destination_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            path = root / 'src/example.py'
            path.parent.mkdir()
            path.symlink_to(root / 'missing')
            with self.assertRaises(ValueError):
                m.restore(root, manifest)
            self.assertTrue(path.is_symlink())

    def test_parent_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            (root / 'different').mkdir()
            (root / 'src').symlink_to(root / 'different', target_is_directory=True)
            with self.assertRaises(ValueError):
                m.restore(root, manifest)
            self.assertFalse((root / 'different/example.py').exists())

    def test_assembly_directory_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            manifest = self.fixture(root)
            (root / 'local/assets/science_snapshot').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                m.restore(root, manifest)
            self.assertEqual(list(Path(outside).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
