"""Storage safety tests use disposable byte fixtures only."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import stream_completed_candidate_checkpoints as archive


class StreamArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.grid = self.root / "grid"
        self.grid.mkdir()
        protocol = {"training": {"policies": ["source", "supported"],
                    "learning_rates": [0.001], "seeds": [17], "epochs": 2}}
        self.write(self.root / "protocol.json", protocol)
        identity = {"protocol": {"path": "protocol.json", "sha256": archive.digest(self.root / "protocol.json")}}
        self.ledger_hash = archive.hashlib.sha256(archive.canonical(identity)).hexdigest()
        self.write(self.grid / "ledger.json", {"identity": identity, "ledger_sha256": self.ledger_hash})
        self.name = "candidates/source/lr_0.001/seed_17"
        self.candidate = self.grid / self.name
        hashes = {}
        for epoch in range(3):
            for name in (f"epochs/epoch_{epoch:02d}.pt", f"validation/epoch_{epoch:02d}.npz"):
                path = self.candidate / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name.encode() + bytes(range(64))) * 40)
                hashes[name] = archive.digest(path)
        self.write(self.candidate / "history.json", {"history": [0, 1, 2]})
        hashes["history.json"] = archive.digest(self.candidate / "history.json")
        self.write(self.candidate / "completion.json", {
            "ledger_sha256": self.ledger_hash, "method": "source", "learning_rate": 0.001,
            "seed": 17, "epochs_completed": 2, "checkpoint_count": 3, "loss_policy": "source",
            "selection_sha256": None, "draw_id": None, "artifact_sha256": hashes})
        self.args = argparse.Namespace(repository=self.root, grid_root="grid", candidate=[self.name],
            output_root=self.root / "archives", prefix="fixture", part_limit_bytes=10000,
            resident_limit_bytes=20000)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.canonical(value) + b"\n")

    def make(self):
        self.public = archive.create(self.args)
        self.manifest = Path(self.public["manifest"]["local_path"])
        self.sha = self.public["manifest"]["sha256"]
        self.index = archive.load_index(self.manifest, self.sha, verify_zips=True)
        self.results = [{**x, "status": "succeeded", "purpose": "create_library_file",
                         "file_id": "fixture-file", "library_file_id": "fixture-library"}
                        for x in [self.public["manifest"], *self.public["parts"]]]
        self.input_receipt = self.root / "input-upload.json"
        self.write(self.input_receipt, {"results": self.results})
        self.upload_args = argparse.Namespace(manifest=self.manifest, manifest_sha256=self.sha,
            upload_receipt=self.input_receipt, output_receipt=self.root / "verified-private-upload.json")

    def accept(self):
        result = archive.record_uploads(self.upload_args)
        self.prune_args = argparse.Namespace(repository=self.root, manifest=self.manifest,
            manifest_sha256=self.sha, upload_record=self.upload_args.output_receipt,
            upload_record_sha256=result["verified_upload_record"]["sha256"],
            confirm_uploaded_and_authorize_deletion=True)

    def finish_selected_fixture(self):
        other = self.grid / "candidates/supported/lr_0.001/seed_17"
        shutil.copytree(self.candidate, other)
        receipt = archive.read_json(other / "completion.json")
        receipt.update(method="supported", loss_policy="supported")
        self.write(other / "completion.json", receipt)
        hashes = {}
        for name in ("primary", "sensitivity"):
            self.write(self.grid / f"selection_{name}.json", {"ledger_sha256": self.ledger_hash})
            hashes[name] = archive.digest(self.grid / f"selection_{name}.json")
        states = [{"state_id": f"{method}_{epoch}",
                   "checkpoint": f"grid/candidates/{method}/lr_0.001/seed_17/epochs/epoch_{epoch:02d}.pt",
                   "checkpoint_sha256": archive.digest(self.grid / f"candidates/{method}/lr_0.001/seed_17/epochs/epoch_{epoch:02d}.pt")}
                  for method in ("source", "supported") for epoch in range(3)]
        full = {"ledger_sha256": self.ledger_hash, "states": states, "selection_sha256": hashes,
                "selections": [{"state_id": "source_1"}],
                "decomposition_selections": [{"state_id": "supported_1"}]}
        self.write(self.grid / "state_manifest.json", full)
        keep = {**full, "states": [s for s in states if not s["state_id"].endswith("_0")],
                "full_state_manifest": {"path": "grid/state_manifest.json",
                                        "sha256": archive.digest(self.grid / "state_manifest.json")}}
        self.write(self.grid / "keep.json", keep)
        self.args.retained_manifest = "grid/keep.json"
        self.args.retained_manifest_sha256 = archive.digest(self.grid / "keep.json")

    def test_selected_grid_two_sequential_batches_keep_selected_states(self):
        self.finish_selected_fixture()
        self.make()
        self.assertEqual(self.public["member_count"], 1)
        self.accept()
        archive.prune(self.prune_args)
        self.assertFalse((self.candidate / "epochs/epoch_00.pt").exists())
        self.assertTrue((self.candidate / "epochs/epoch_01.pt").exists())
        self.assertTrue((self.candidate / "epochs/epoch_02.pt").exists())
        for part in self.public["parts"]:
            Path(part["local_path"]).unlink()
        self.args.prefix = "second"
        self.args.candidate = ["candidates/supported/lr_0.001/seed_17"]
        self.upload_args.output_receipt.unlink()
        # This step must work after the first candidate's archived weights disappear.
        self.make()
        self.accept()
        outcome = archive.prune(self.prune_args)
        self.assertEqual(outcome["deleted_count"], 1)
        for method in ("source", "supported"):
            directory = self.grid / f"candidates/{method}/lr_0.001/seed_17/epochs"
            self.assertFalse((directory / "epoch_00.pt").exists())
            self.assertTrue((directory / "epoch_01.pt").exists())
            self.assertTrue((directory / "epoch_02.pt").exists())

    def test_retained_manifest_cannot_omit_selection_or_terminal(self):
        self.finish_selected_fixture()
        path = self.grid / "keep.json"
        original = archive.read_json(path)
        for sid, message in (("source_1", "selected/decomposition"),
                             ("supported_1", "selected/decomposition"),
                             ("source_2", "terminal checkpoint")):
            value = {**original, "states": [s for s in original["states"] if s["state_id"] != sid]}
            self.write(path, value)
            self.args.retained_manifest_sha256 = archive.digest(path)
            with self.assertRaisesRegex(ValueError, message):
                archive.create(self.args)

    def test_retained_manifest_requires_both_selection_hashes(self):
        self.finish_selected_fixture()
        path = self.grid / "keep.json"
        keep = archive.read_json(path)
        del keep["selection_sha256"]["sensitivity"]
        self.write(path, keep)
        self.args.retained_manifest_sha256 = archive.digest(path)
        with self.assertRaisesRegex(ValueError, "Both frozen selection"):
            archive.create(self.args)

    def test_explicit_factorial_selection_must_be_retained(self):
        self.finish_selected_fixture()
        path = self.grid / "keep.json"
        keep = archive.read_json(path)
        keep["factorial_selections"] = [{"state_id": "source_0"}]
        self.write(path, keep)
        self.args.retained_manifest_sha256 = archive.digest(path)
        with self.assertRaisesRegex(ValueError, "factorial state"):
            archive.create(self.args)

    def test_changed_selected_state_blocks_later_batch(self):
        self.finish_selected_fixture()
        (self.grid / "candidates/supported/lr_0.001/seed_17/epochs/epoch_01.pt").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            archive.create(self.args)

    def test_both_frozen_matched_control_schemas(self):
        spec = importlib.util.spec_from_file_location("old_archive_tests", Path(__file__).with_name("test_archive_completed_grid_checkpoints.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for schema in ("ad", "retention"):
            with self.subTest(schema=schema):
                fixture = module.ArchiveTests()
                fixture.setUp()
                try:
                    fixture.fixture(schema)
                    args = argparse.Namespace(repository=fixture.root, grid_root="grid",
                        candidate=["candidates/supported/lr_0.001/seed_17"],
                        output_root=fixture.root / "bounded", prefix="bounded", part_limit_bytes=10000,
                        resident_limit_bytes=20000, retained_manifest=fixture.args.retained_manifest,
                        retained_manifest_sha256=fixture.args.retained_manifest_sha256)
                    outcome = archive.create(args)
                    self.assertEqual(outcome["member_count"], 2)
                    args.prefix = "matched"
                    prefix = "matched_allocation_distillation_draw_" if schema == "ad" else "matched_distilled_draw_"
                    args.candidate = [f"candidates/{prefix}0/lr_0.001/seed_17"]
                    outcome = archive.create(args)
                    self.assertEqual(outcome["member_count"], 1)
                finally:
                    fixture.doCleanups()

    def test_partial_grid_round_trip_without_resident_zips(self):
        original = {p.relative_to(self.root).as_posix(): p.read_bytes()
                    for p in self.root.rglob("*") if p.is_file()}
        self.make()
        self.assertEqual(self.public["member_count"], 2)
        self.assertFalse((self.grid / "candidates/supported").exists())
        self.accept()
        part = Path(self.public["parts"][0]["local_path"])
        remote = self.root / "remote" / part.name
        remote.parent.mkdir()
        part.rename(remote)
        outcome = archive.prune(self.prune_args)
        self.assertEqual(outcome["deleted_count"], 2)
        self.assertFalse((self.candidate / "epochs/epoch_00.pt").exists())
        self.assertTrue((self.candidate / "epochs/epoch_02.pt").exists())
        restore_args = argparse.Namespace(repository=self.root, manifest=self.manifest,
                                         manifest_sha256=self.sha, part=remote)
        self.assertEqual(archive.restore(restore_args)["restored_count"], 2)
        self.assertEqual(archive.restore(restore_args)["restored_count"], 0)
        self.assertTrue(all((self.root / name).read_bytes() == value for name, value in original.items()))
        self.assertEqual(archive.inspect_candidates(self.root, "grid", [self.name]), self.index["snapshot"])

    def test_missing_or_failed_upload_rejected(self):
        self.make()
        for results in (self.results[:1], [{**r, "status": "failed"} for r in self.results]):
            self.write(self.input_receipt, {"results": results})
            with self.assertRaises(ValueError):
                archive.record_uploads(self.upload_args)
        self.assertTrue((self.candidate / "epochs/epoch_00.pt").exists())

    def test_stale_duplicate_or_missing_ids_rejected(self):
        self.make()
        variants = [
            [{**self.results[0], "sha256": "0" * 64}, self.results[1]],
            [self.results[0], self.results[0]],
            [{**r, "file_id": None} for r in self.results],
        ]
        for results in variants:
            self.write(self.input_receipt, {"results": results})
            with self.assertRaises(ValueError):
                archive.record_uploads(self.upload_args)

    def test_upload_acceptance_requires_local_archive(self):
        self.make()
        Path(self.public["parts"][0]["local_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing regular ZIP"):
            archive.record_uploads(self.upload_args)

    def test_uploaded_part_corruption_rejected(self):
        self.make()
        Path(self.public["parts"][0]["local_path"]).write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "digest/size"):
            archive.record_uploads(self.upload_args)

    def test_source_corruption_blocks_prune(self):
        self.make()
        self.accept()
        (self.candidate / "epochs/epoch_00.pt").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            archive.prune(self.prune_args)
        self.assertTrue((self.candidate / "epochs/epoch_01.pt").exists())
        self.assertFalse(list(self.args.output_root.glob("*PRUNE*")))

    def test_retained_or_validation_corruption_blocks_prune(self):
        for name in ("epochs/epoch_02.pt", "validation/epoch_01.npz"):
            with self.subTest(name=name):
                self.make()
                self.accept()
                path = self.candidate / name
                old = path.read_bytes()
                path.write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    archive.prune(self.prune_args)
                path.write_bytes(old)
                self.assertTrue((self.candidate / "epochs/epoch_00.pt").exists())
                # Fresh independent archive namespace for the next fixture check.
                self.args.prefix += "x"
                self.upload_args.output_receipt.unlink()

    def test_no_prune_without_explicit_authorization(self):
        self.make()
        self.accept()
        self.prune_args.confirm_uploaded_and_authorize_deletion = False
        with self.assertRaisesRegex(ValueError, "authorization"):
            archive.prune(self.prune_args)

    def test_changed_upload_record_blocks_prune(self):
        self.make()
        self.accept()
        self.upload_args.output_receipt.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "record digest"):
            archive.prune(self.prune_args)

    def test_overlapping_or_omitted_members_rejected(self):
        self.make()
        original = self.manifest.read_bytes()
        self.index["parts"][0]["members"].append(self.index["parts"][0]["members"][0])
        self.write(self.manifest, self.index)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            archive.load_index(self.manifest, archive.digest(self.manifest))
        self.manifest.write_bytes(original)
        index = archive.read_json(self.manifest)
        index["parts"][0]["members"].pop()
        self.write(self.manifest, index)
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            archive.load_index(self.manifest, archive.digest(self.manifest))

    def test_resident_limit_counts_previous_partial_files(self):
        self.args.output_root.mkdir()
        (self.args.output_root / "old.zip.partial").write_bytes(b"x" * 15000)
        with self.assertRaisesRegex(ValueError, "Resident archive bound"):
            archive.create(self.args)
        self.assertFalse(list(self.args.output_root.glob("fixture*")))
        self.assertTrue((self.candidate / "epochs/epoch_00.pt").exists())

    def test_interrupted_prefix_is_not_overwritten(self):
        self.args.output_root.mkdir()
        (self.args.output_root / "fixture_part_0001.zip.partial").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "Prefix already exists"):
            archive.create(self.args)

    def test_upload_record_is_immutable(self):
        self.make()
        self.accept()
        with self.assertRaises(FileExistsError):
            archive.record_uploads(self.upload_args)

    def test_interrupted_prune_is_not_repeated(self):
        self.make()
        self.accept()
        self.write(self.args.output_root / "fixture_LOCAL_PRUNE_RECEIPT.json", {"status": "in_progress"})
        with self.assertRaisesRegex(ValueError, "Prune receipt exists"):
            archive.prune(self.prune_args)

    def test_corrupt_restore_never_overwrites_original(self):
        self.make()
        (self.candidate / "epochs/epoch_00.pt").write_bytes(b"existing-different")
        args = argparse.Namespace(repository=self.root, manifest=self.manifest, manifest_sha256=self.sha,
                                  part=Path(self.public["parts"][0]["local_path"]))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            archive.restore(args)
        self.assertEqual((self.candidate / "epochs/epoch_00.pt").read_bytes(), b"existing-different")

    def test_frozen_selection_blocks_preselection_archive(self):
        self.write(self.grid / "selection_primary.json", {})
        with self.assertRaisesRegex(ValueError, "before selection"):
            archive.create(self.args)

    def test_candidate_busy_blocks_archive(self):
        with (self.candidate / ".candidate.lock").open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "busy"):
                archive.create(self.args)

    def test_unknown_weight_missing_artifact_and_symlink_rejected(self):
        extra = self.candidate / "epochs/epoch_99.pt"
        extra.write_bytes(b"unbound")
        with self.assertRaisesRegex(ValueError, "Unbound"):
            archive.create(self.args)
        extra.unlink()
        path = self.candidate / "epochs/epoch_00.pt"
        external = self.root / "saved.pt"
        path.rename(external)
        with self.assertRaisesRegex(ValueError, "missing candidate weights"):
            archive.create(self.args)
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            archive.create(self.args)


if __name__ == "__main__":
    unittest.main()
