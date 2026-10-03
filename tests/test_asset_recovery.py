"""Exercise exact bytes, incomplete/corrupt parts, and safe publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_asset_part import MIB, package_part  # noqa: E402
from restore_assets import restore_assets  # noqa: E402


class AssetRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="asset-recovery-test-", dir=ROOT.parent)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.relative = "results/features/sample/features.npz"
        self.original = os.urandom(2 * MIB + 113)
        path = self.source / self.relative
        path.parent.mkdir(parents=True)
        path.write_bytes(self.original)
        (path.parent / "metadata.json").write_text('{"example": true}\n')
        self.recovery = self.base / "recovery"
        self.recovery.mkdir()
        self.archives = []
        self.receipts = []
        for index in range(3):
            output = self.base / f"part-{index}.zip"
            receipt = package_part(self.source, self.relative, index, MIB, output,
                                   ["results/features/sample/metadata.json"])
            self.archives.append(output)
            self.receipts.append(receipt)
        self.asset_id = self.receipts[0]["asset_id"]
        self.target = self.recovery / "project" / self.relative

    def tearDown(self):
        self.temporary.cleanup()

    def extract(self, index):
        with zipfile.ZipFile(self.archives[index]) as archive:
            archive.extractall(self.recovery)

    def test_missing_corrupt_and_complete_exact_bytes(self):
        self.extract(0)
        self.extract(1)
        report, code = restore_assets(self.recovery)
        self.assertEqual(code, 2)
        self.assertFalse(report["complete"])
        self.assertEqual(report["assets"][0]["missing_part_indices"], [2])
        self.assertFalse(self.target.exists())
        self.extract(2)
        last = self.recovery / "ASSET_PARTS" / self.asset_id / "000002.bin"
        correct = last.read_bytes()
        last.write_bytes(b"x" + correct[1:])
        report, code = restore_assets(self.recovery)
        self.assertEqual(code, 1)
        self.assertEqual(report["assets"][0]["status"], "corrupt_chunk")
        self.assertFalse(self.target.exists())
        last.write_bytes(correct)
        # Run the standalone helper copied into the actual received ZIP.
        result = subprocess.run([sys.executable, str(self.recovery / "restore_assets.py"), "--root", str(self.recovery)],
                                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(json.loads(result.stdout)["complete"])
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertEqual((self.source / self.relative).read_bytes(), self.original)
        descriptor_bytes = []
        for archive in self.archives:
            with zipfile.ZipFile(archive) as z:
                descriptor_bytes.append(z.read(f"ASSET_DESCRIPTORS/{self.asset_id}.json"))
        self.assertEqual(len(set(descriptor_bytes)), 1)

    def test_existing_target_preserved_until_explicit_verified_overwrite(self):
        for index in range(3):
            self.extract(index)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(b"keep this existing file")
        report, code = restore_assets(self.recovery)
        self.assertEqual(code, 1)
        self.assertEqual(report["assets"][0]["status"], "existing_conflict")
        self.assertEqual(self.target.read_bytes(), b"keep this existing file")
        chunk = self.recovery / "ASSET_PARTS" / self.asset_id / "000000.bin"
        correct = chunk.read_bytes()
        chunk.write_bytes(b"corrupt")
        report, code = restore_assets(self.recovery, overwrite=True)
        self.assertEqual(code, 1)
        self.assertEqual(self.target.read_bytes(), b"keep this existing file")
        chunk.write_bytes(correct)
        report, code = restore_assets(self.recovery, overwrite=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.target.read_bytes(), self.original)
        report, code = restore_assets(self.recovery)
        self.assertEqual(report["assets"][0]["status"], "already_present")
        self.assertEqual(code, 0)

    def test_multiple_assets_missing_and_conflicting_versions(self):
        for index in range(3):
            self.extract(index)
        second_relative = "checkpoints/other.pt"
        second = self.source / second_relative
        second.parent.mkdir(parents=True)
        second.write_bytes(os.urandom(MIB + 9))
        archive = self.base / "other-part-0.zip"
        package_part(self.source, second_relative, 0, MIB, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(self.recovery)
        report, code = restore_assets(self.recovery)
        self.assertEqual(code, 2)
        self.assertFalse(report["complete"])
        self.assertEqual(self.target.read_bytes(), self.original)
        archive2 = self.base / "other-part-1.zip"
        package_part(self.source, second_relative, 1, MIB, archive2)
        with zipfile.ZipFile(archive2) as z:
            z.extractall(self.recovery)
        report, code = restore_assets(self.recovery)
        self.assertEqual(code, 0)
        self.assertEqual((self.recovery / "project" / second_relative).read_bytes(), second.read_bytes())
        # An alternate version of the same target needs explicit version choice.
        second.write_bytes(b"alternate version")
        other_version_zip = self.base / "other-version.zip"
        other_receipt = package_part(self.source, second_relative, 0, MIB, other_version_zip)
        with zipfile.ZipFile(other_version_zip) as z:
            z.extractall(self.recovery)
        report, code = restore_assets(self.recovery, overwrite=True)
        self.assertEqual(code, 1)
        self.assertIn("conflicting_descriptors", report["counts"])
        report, code = restore_assets(self.recovery, [other_receipt["asset_id"]], overwrite=True)
        self.assertEqual(code, 0)
        self.assertEqual((self.recovery / "project" / second_relative).read_bytes(), b"alternate version")

    def test_immutable_deterministic_outputs_and_invalid_sources(self):
        before = hashlib.sha256(self.archives[0].read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            package_part(self.source, self.relative, 0, MIB, self.archives[0])
        self.assertEqual(hashlib.sha256(self.archives[0].read_bytes()).hexdigest(), before)
        duplicate = self.base / "same-content.zip"
        package_part(self.source, self.relative, 0, MIB, duplicate, ["results/features/sample/metadata.json"])
        self.assertEqual(duplicate.read_bytes(), self.archives[0].read_bytes())
        with self.assertRaises(ValueError):
            package_part(self.source, "../outside.bin", 0, MIB, self.base / "unsafe.zip")
        (self.source / ".env").write_text("example secret")
        with self.assertRaises(ValueError):
            package_part(self.source, ".env", 0, MIB, self.base / "secret.zip")
        outside = self.base / "outside.bin"
        outside.write_bytes(b"outside")
        (self.source / "escape.bin").symlink_to(outside)
        with self.assertRaises(ValueError):
            package_part(self.source, "escape.bin", 0, MIB, self.base / "escape.zip")
        with self.assertRaises(ValueError):
            package_part(self.source, self.relative, 3, MIB, self.base / "bad-index.zip")

    def test_mismatched_descriptor_and_part_path_are_rejected(self):
        for index in range(3):
            self.extract(index)
        descriptor_path = self.recovery / "ASSET_DESCRIPTORS" / f"{self.asset_id}.json"
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["chunks"][0]["path"] = "../outside.bin"
        descriptor_path.write_text(json.dumps(descriptor))
        report, code = restore_assets(self.recovery)
        self.assertEqual(code, 1)
        self.assertEqual(report["assets"][0]["status"], "invalid_descriptor")
        self.assertFalse(self.target.exists())

    def test_raw_payload_cap_is_enforced_before_publication(self):
        large_include = self.source / "oversized_metadata.bin"
        large_include.write_bytes(b"\0" * (34 * MIB))
        output = self.base / "over-cap.zip"
        with self.assertRaisesRegex(ValueError, "exceeding 35 MiB"):
            package_part(self.source, self.relative, 0, MIB, output, ["oversized_metadata.bin"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
