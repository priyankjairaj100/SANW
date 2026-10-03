"""A valid ZIP CRC is insufficient: verify manifest completeness and hashes."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_recovery import verify  # noqa: E402


class RecoveryVerifierTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recovery-verifier-test-", dir=ROOT.parent)
        self.base = Path(self.temporary.name)
        self.payloads = {"project/data.bin": b"exact binary bytes\x00\xff", "project/results.json": b'{"score": 1}\n'}
        self.manifest = {"files": [{"path": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                                   for name, value in self.payloads.items()]}

    def tearDown(self):
        self.temporary.cleanup()

    def make_zip(self, name, payloads):
        path = self.base / name
        with zipfile.ZipFile(path, "w") as archive:
            for key, value in payloads.items():
                archive.writestr(key, value)
            archive.writestr("RECOVERY_MANIFEST.json", json.dumps(self.manifest))
        return path

    def test_zip_hashes_missing_payload_and_unlisted_files(self):
        good = self.make_zip("good.zip", self.payloads)
        report, code = verify(archive_path=good)
        self.assertEqual(code, 0)
        self.assertTrue(report["complete"])
        self.assertTrue(report["zip_crc_verified"])
        tampered = dict(self.payloads, **{"project/data.bin": b"valid ZIP, wrong file"})
        report, code = verify(archive_path=self.make_zip("tampered.zip", tampered))
        self.assertEqual(code, 1)
        self.assertEqual(report["failures"][0]["kind"], "content_mismatch")
        incomplete = {"project/data.bin": self.payloads["project/data.bin"]}
        report, code = verify(archive_path=self.make_zip("missing.zip", incomplete))
        self.assertEqual(code, 1)
        self.assertEqual(report["missing_files"], ["project/results.json"])
        extra = dict(self.payloads, **{"project/unlisted.bin": b"not inventoried"})
        extra_zip = self.make_zip("extra.zip", extra)
        report, code = verify(archive_path=extra_zip)
        self.assertEqual(code, 1)
        report, code = verify(archive_path=extra_zip, allow_unlisted=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["unlisted_files"], ["project/unlisted.bin"])

    def test_extracted_files_and_unsafe_names(self):
        archive = self.make_zip("good.zip", self.payloads)
        extracted = self.base / "extracted"
        with zipfile.ZipFile(archive) as z:
            z.extractall(extracted)
        report, code = verify(root=extracted)
        self.assertEqual(code, 0)
        (extracted / "project/data.bin").write_bytes(b"changed")
        report, code = verify(root=extracted)
        self.assertEqual(code, 1)
        bad = dict(self.payloads, **{"../outside.bin": b"unsafe"})
        report, code = verify(archive_path=self.make_zip("unsafe.zip", bad))
        self.assertEqual(code, 1)
        self.assertIn("Unsafe archive path", report["error"])


if __name__ == "__main__":
    unittest.main()
