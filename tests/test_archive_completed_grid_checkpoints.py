"""Safety checks use disposable byte fixtures, never research checkpoints."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("archive_helper", Path(__file__).parents[1] / "scripts/archive_completed_grid_checkpoints.py")
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.grid = self.root / "grid"
        self.grid.mkdir()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.canonical(value) + b"\n")

    def fixture(self, control="ad"):
        training = {"policies": ["supported"], "learning_rates": [0.001], "seeds": [17], "epochs": 2}
        family = "allocation_distillation" if control == "ad" else "distilled"
        prefix = "matched_allocation_distillation_draw_" if control == "ad" else "matched_distilled_draw_"
        if control == "ad":
            training["matched_controls"] = {"fits_per_encoder": 3}
        else:
            training["matched_distilled_control"] = {"additional_runs_per_encoder": 3}
        self.write(self.root / "protocol.json", {"training": training})
        identity = {"protocol": {"path": "protocol.json", "sha256": archive.digest(self.root / "protocol.json")}}
        self.ledger_hash = archive.hashlib.sha256(archive.canonical(identity)).hexdigest()
        self.write(self.grid / "ledger.json", {"identity": identity, "ledger_sha256": self.ledger_hash})
        selection = {"ledger_sha256": self.ledger_hash,
                     "families": {family: {"policy": "supported", "learning_rate": 0.001,
                                            "runs": [{"seed": 17, "epoch": 1}]}}}
        selection_hashes = {}
        for name in ("primary", "sensitivity"):
            self.write(self.grid / f"selection_{name}.json", selection)
            selection_hashes[name] = archive.digest(self.grid / f"selection_{name}.json")
        states, selected = [], []
        for method, n, draw in [("supported", 2, None)] + [(prefix + str(draw), 1, draw) for draw in range(3)]:
            directory = self.grid / "candidates" / method / "lr_0.001" / "seed_17"
            hashes = {}
            for epoch in range(n + 1):
                for name in (f"epochs/epoch_{epoch:02d}.pt", f"validation/epoch_{epoch:02d}.npz"):
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((method + name).encode() * 40)
                    hashes[name] = archive.digest(path)
                state = {"state_id": f"{method}_{epoch}", "checkpoint": (directory / f"epochs/epoch_{epoch:02d}.pt").relative_to(self.root).as_posix(),
                         "checkpoint_sha256": hashes[f"epochs/epoch_{epoch:02d}.pt"]}
                states.append(state)
                if epoch == n:
                    selected.append(state)
            self.write(directory / "history.json", {"history": list(range(n + 1))})
            hashes["history.json"] = archive.digest(directory / "history.json")
            self.write(directory / "completion.json", {"ledger_sha256": self.ledger_hash,
                "method": method, "seed": 17, "learning_rate": 0.001, "epochs_completed": n,
                "checkpoint_count": n + 1, "loss_policy": "supported", "draw_id": draw,
                "selection_sha256": selection_hashes["primary"] if draw is not None else None,
                "artifact_sha256": hashes})
        full = {"ledger_sha256": self.ledger_hash, "states": states, "selection_sha256": selection_hashes,
                "selections": [{"state_id": state["state_id"]} for state in selected], "matched_controls_complete": True}
        self.write(self.grid / "state_manifest.json", full)
        keep = {**full, "states": selected, "full_state_manifest": {
            "path": "grid/state_manifest.json", "sha256": archive.digest(self.grid / "state_manifest.json")}}
        self.write(self.grid / "archival_keep_manifest.json", keep)
        self.args = argparse.Namespace(repository=self.root, grid_root="grid",
            retained_manifest="grid/archival_keep_manifest.json",
            retained_manifest_sha256=archive.digest(self.grid / "archival_keep_manifest.json"),
            expected_completions=4, output_root=self.root / "archives", prefix="fixture", part_limit_bytes=5000)
        return states, selected

    def inspect(self):
        return archive.inspect_grid(self.root, "grid", self.args.retained_manifest,
                                    self.args.retained_manifest_sha256, 4)

    def test_ad_lossless_parts_and_prune(self):
        states, selected = self.fixture()
        original = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        public = archive.archive_grid(self.args)
        self.assertEqual(public["archived_members"], 5)
        self.assertGreater(len(public["parts"]), 1)
        self.assertTrue(all(p["uncompressed_bytes"] <= 5000 for p in public["parts"]))
        self.assertTrue(all((self.root / name).read_bytes() == content for name, content in original.items()))
        manifest = self.args.output_root / public["manifest"]["name"]
        index = archive.verify_index(manifest, public["manifest"]["sha256"])
        results = [{"local_path": str(self.args.output_root / item["name"]), "bytes": item["bytes"],
                    "sha256": item["sha256"], "status": "succeeded", "purpose": "create_library_file",
                    "file_id": "test", "library_file_id": "test"}
                   for item in [public["manifest"], *public["parts"]]]
        receipt = self.root / "upload.json"
        self.write(receipt, {"results": results})
        prune = argparse.Namespace(repository=self.root, manifest=manifest, manifest_sha256=public["manifest"]["sha256"],
            upload_receipt=receipt, confirm_uploaded_and_authorize_deletion=True)
        # A stale success for different uploaded bytes must fail before deletion.
        results[0]["sha256"] = "0" * 64
        self.write(receipt, {"results": results})
        with self.assertRaisesRegex(ValueError, "uploaded source digest"):
            archive.prune_grid(prune)
        self.assertTrue(all((self.root / state["checkpoint"]).exists() for state in states))
        results[0]["sha256"] = public["manifest"]["sha256"]
        self.write(receipt, {"results": results})
        outcome = archive.prune_grid(prune)
        self.assertEqual(outcome["deleted_count"], 5)
        for item in index["snapshot"]["protected_files"]:
            self.assertEqual((self.root / item["path"]).read_bytes(), original[item["path"]])
        archive.verify_index(manifest, public["manifest"]["sha256"])

    def test_retention_control_schema(self):
        self.fixture("retention")
        self.assertEqual(len(self.inspect()["members"]), 5)

    def test_full_state_manifest_retains_every_epoch(self):
        self.fixture()
        self.args.retained_manifest = "grid/state_manifest.json"
        self.args.retained_manifest_sha256 = archive.digest(self.grid / "state_manifest.json")
        self.assertEqual(self.inspect()["members"], [])

    def test_missing_completion_rejected(self):
        self.fixture()
        next(self.grid.rglob("completion.json")).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete/unexpected"):
            self.inspect()

    def test_nonweight_corruption_rejected(self):
        self.fixture()
        next(self.grid.rglob("*.npz")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.inspect()

    def test_unbound_weight_rejected(self):
        self.fixture()
        (next(self.grid.rglob("epochs")) / "epoch_99.pt").write_bytes(b"unbound")
        with self.assertRaisesRegex(ValueError, "Unbound or missing"):
            self.inspect()

    def test_missing_selected_state_rejected(self):
        self.fixture()
        path = self.grid / "archival_keep_manifest.json"
        keep = archive.read_json(path)
        keep["states"].pop()
        self.write(path, keep)
        self.args.retained_manifest_sha256 = archive.digest(path)
        with self.assertRaisesRegex(ValueError, "selected/decomposition state"):
            self.inspect()

    def test_symlink_rejected(self):
        self.fixture()
        path = next(self.grid.rglob("*.pt"))
        saved = self.root / "external.pt"
        path.rename(saved)
        path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
