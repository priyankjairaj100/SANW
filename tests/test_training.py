"""Selection, leakage barriers, deterministic fitting and recovery checks."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gcr.adapters import ResidualAdapter
from gcr.training import (
    FeatureDataset, METHODS, StudyConfig, atomic_json, atomic_torch_save,
    candidate_directory, ensure_ledger, fit_candidate, select_candidates,
    sha256_file, validate_adapter, make_ledger,
)


def fixture_dataset(root: Path) -> FeatureDataset:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "dataset": "synthetic_test_only", "images": [], "texts": [], "pairs": []}
    image_features, text_features = [], []
    for index, split in enumerate(("train", "train", "validation", "validation", "test")):
        iid = f"image:{index}"
        vector = np.eye(8, dtype=np.float32)[index]
        manifest["images"].append({"id": iid, "split": split, "path": f"unused/{index}.jpg"})
        image_features.append(vector)
        # Source is imperfect; support is the exact image vector. This catches
        # accidental source-only validation relevance, which was not selected.
        source = vector + .3 * np.eye(8, dtype=np.float32)[7]
        source /= np.linalg.norm(source)
        for relation, feature in (("source", source), ("supported", vector), ("contradicted", -vector), ("neutral", np.eye(8, dtype=np.float32)[6])):
            tid = f"text:{index}:{relation}"
            manifest["texts"].append({"id": tid, "text": tid})
            manifest["pairs"].append({"image_id": iid, "text_id": tid, "relation": relation})
            text_features.append(feature)
    atomic_json(root / "manifest.json", manifest)
    np.savez(root / "features.npz", image_features=image_features, text_features=text_features,
             image_ids=[x["id"] for x in manifest["images"]], text_ids=[x["id"] for x in manifest["texts"]])
    atomic_json(root / "metadata.json", {"manifest_sha256": sha256_file(root / "manifest.json"), "logit_scale": 10.0})
    return FeatureDataset(root, root / "manifest.json", root / "features.npz", root / "metadata.json", dimension=8)


def test_validation_uses_supported_targets_and_excludes_test_candidates(tmp_path):
    data = fixture_dataset(tmp_path)
    metrics = validate_adapter(ResidualAdapter(8), data)
    assert metrics["score"] == metrics["known_positive_i2t_r1"] == metrics["relation_accuracy"] == 1
    assert metrics["image_count"] == 2 and metrics["text_count"] == 8
    assert [x["winning_text_id"] for x in metrics["per_image"]] == ["text:2:supported", "text:3:supported"]
    assert all(not x["image_id"].endswith(":4") for x in metrics["per_image"])
    # Arbitrary held-out feature changes cannot alter checkpoint validation.
    data.images[4] *= -1
    data.texts[16:] *= -1
    assert validate_adapter(ResidualAdapter(8), data) == metrics


def test_batch_cross_image_pairs_remain_unannotated(tmp_path):
    data = fixture_dataset(tmp_path)
    _, _, relations, text_indices = data.batch([0, 1])
    assert text_indices == list(range(8))
    assert relations.tolist() == [[1, 2, 3, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 3, 4]]


def test_manifest_binding_rejects_wrong_digest(tmp_path):
    data = fixture_dataset(tmp_path)
    atomic_json(tmp_path / "metadata.json", {"manifest_sha256": "bad", "logit_scale": 10})
    with pytest.raises(ValueError, match="exact manifest"):
        FeatureDataset(tmp_path, data.manifest_path, data.features_path, data.metadata_path, 8)


def test_immutable_ledger_rejects_changed_execution(tmp_path):
    ledger = {"identity": {"protocol": "first"}, "ledger_sha256": "a"}
    ensure_ledger(tmp_path, ledger)
    ensure_ledger(tmp_path, ledger)
    with pytest.raises(ValueError, match="immutable"):
        ensure_ledger(tmp_path, {"identity": {"protocol": "second"}, "ledger_sha256": "b"})


def protocol_fixture(config):
    return {
        "training": {
            "methods": list(config.methods), "learning_rates": list(config.learning_rates),
            "seeds": list(config.seeds), "epochs": config.epochs,
            "image_batch_size": config.image_batch_size, "weight_decay": config.weight_decay,
            "betas": list(config.adam_betas), "epsilon": config.adam_epsilon,
            "amsgrad": config.adam_amsgrad, "foreach": config.adam_foreach,
            "fused": config.adam_fused, "gradient_clip_norm": config.grad_clip_norm,
            "optimizer": config.optimizer, "device": config.device,
        },
        "encoder": {"feature_dim": config.feature_dim},
        "adapter": {"dimension": config.feature_dim},
        "objectives": {
            "fixed_threshold": config.semantic_threshold, "semantic_softness": config.semantic_softness,
            "constant_negative_weight": config.negative_weight, "contradiction_weight": config.hardening_weight,
            "smoothing_coefficient": config.smoothing,
        },
        "selection": {
            "positive_relations": list(config.validation_positives), "retrieval_tie_rule": config.validation_tie_rule,
            "hypothesis_ties": config.relation_tie_credit, "tie_rule": config.selection_tie_rule,
        },
    }


@pytest.mark.parametrize("changed", [
    {"learning_rates": (1e-4, 1e-3)}, {"seeds": (17, 29)}, {"epochs": 1},
    {"weight_decay": .02}, {"smoothing": .2}, {"validation_positives": ("source",)},
])
def test_ledger_rejects_scientific_config_change_even_with_correct_protocol_hash(tmp_path, changed):
    data = fixture_dataset(tmp_path)
    config = StudyConfig(feature_dim=8, threads=1)
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, protocol_fixture(config))
    for source in ("src/gcr/training.py", "src/gcr/adapters.py", "src/gcr/losses.py", "scripts/run_study.py"):
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic source marker for ledger tests.\n")
    digest = sha256_file(protocol)
    original = make_ledger(tmp_path, data, config, protocol, digest)
    with pytest.raises(ValueError, match="Frozen protocol disagrees"):
        make_ledger(tmp_path, data, dataclasses.replace(config, **changed), protocol, digest)
    # Runtime thread count may differ, but the resulting ledger must differ.
    runtime_change = make_ledger(tmp_path, data, dataclasses.replace(config, threads=2), protocol, digest)
    assert original["ledger_sha256"] != runtime_change["ledger_sha256"]


def test_candidate_repeatability_complete_resume_and_tamper_rejection(tmp_path):
    data = fixture_dataset(tmp_path / "inputs")
    config = StudyConfig(feature_dim=8, epochs=2, threads=1, image_batch_size=1)
    ledger = {"ledger_sha256": "synthetic-test-no-real-study"}
    a = fit_candidate(tmp_path, tmp_path / "a", data, config, ledger, "random_exclusion", 1e-3, 17)
    b = fit_candidate(tmp_path, tmp_path / "b", data, config, ledger, "random_exclusion", 1e-3, 17)
    assert a["best_epoch"] == 0  # exact validation ties must keep epoch zero
    path_a = candidate_directory(tmp_path / "a", "random_exclusion", 1e-3, 17)
    path_b = candidate_directory(tmp_path / "b", "random_exclusion", 1e-3, 17)
    history_a, history_b = [json.loads((path / "history.json").read_text())["history"] for path in (path_a, path_b)]
    for row_a, row_b in zip(history_a, history_b):
        row_a.pop("seconds"); row_b.pop("seconds")
        assert row_a == row_b
    skipped = fit_candidate(tmp_path, tmp_path / "a", data, config, ledger, "random_exclusion", 1e-3, 17)
    assert skipped["skipped_compatible_complete"] is True
    (path_a / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        fit_candidate(tmp_path, tmp_path / "a", data, config, ledger, "random_exclusion", 1e-3, 17)


def test_selection_requires_complete_grid_and_uses_shared_low_rate_tie(tmp_path):
    config = StudyConfig(feature_dim=8, epochs=0, threads=1)
    ledger = {"ledger_sha256": "synthetic-ledger"}
    for method in METHODS:
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_directory(tmp_path / "study", method, rate, seed)
                # Highest-rate scores differ by seed but have mean .5; it must
                # not choose one independent learning rate for each seed.
                value = (0.2, 0.5, 0.8)[config.seeds.index(seed)] if rate == config.learning_rates[-1] else .5
                checkpoint = {"schema_version": 1, "state_dict": ResidualAdapter(8).state_dict(), "ledger_sha256": ledger["ledger_sha256"], "method": method, "learning_rate": rate, "seed": seed, "epoch": 0, "validation": {"score": value}}
                atomic_torch_save(directory / "best.pt", checkpoint)
                atomic_json(directory / "history.json", {"ledger_sha256": ledger["ledger_sha256"]})
                atomic_json(directory / "completion.json", {"ledger_sha256": ledger["ledger_sha256"], "method": method, "learning_rate": rate, "seed": seed, "epochs_completed": 0, "best_validation_score": value, "artifacts_sha256": {name: sha256_file(directory / name) for name in ("best.pt", "history.json")}})
    selection = select_candidates(tmp_path, tmp_path / "study", config, ledger)
    assert selection["selected_checkpoint_count"] == 36
    assert all(row["learning_rate"] == 1e-4 for row in selection["methods"].values())
    candidate_directory(tmp_path / "study", "clip", 1e-3, 43).joinpath("completion.json").unlink()
    with pytest.raises(ValueError, match="complete declared grid"):
        select_candidates(tmp_path, tmp_path / "study", config, ledger)
