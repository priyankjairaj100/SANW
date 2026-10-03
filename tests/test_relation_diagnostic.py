"""Leakage, calibration and threshold checks on constructed data only."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import softmax

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "relation_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("relation_diagnostic", MODULE_PATH)
rd = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rd
SPEC.loader.exec_module(rd)


def synthetic_dataset():
    rng = np.random.default_rng(91)
    images, texts, labels, image_ids, text_ids, splits = [], [], [], [], [], []
    for split, count in (("train", 9), ("validation", 3), ("calibration", 6), ("test", 3)):
        for image in range(count):
            for label in range(3):
                iv = rng.normal(size=3)
                tv = np.eye(3)[label] + rng.normal(scale=.05, size=3)
                images.append(iv / np.linalg.norm(iv))
                texts.append(tv / np.linalg.norm(tv))
                labels.append(label)
                image_ids.append(f"{split}:{image}")
                text_ids.append(f"{split}:{image}:{label}")
                splits.append(split)
    return rd.RelationDataset(np.asarray(images), np.asarray(texts), np.asarray(labels), np.asarray(image_ids),
                              np.asarray(text_ids), np.asarray(splits), np.arange(len(labels)))


def test_calibration_halves_are_reproducible_and_image_disjoint():
    ids = np.repeat(np.array(["e", "b", "d", "a", "c"]), [1, 2, 3, 2, 1])
    temp, gate = rd.split_calibration_images(ids)
    same_temp, same_gate = rd.split_calibration_images(ids)
    assert np.array_equal(temp, same_temp) and np.array_equal(gate, same_gate)
    rd.assert_group_disjoint(ids[temp], ids[gate])
    assert len(np.unique(ids[temp])) == 2 and len(np.unique(ids[gate])) == 3
    assert sorted(np.r_[temp, gate]) == list(range(len(ids)))


def test_crossfit_never_splits_images_and_covers_every_row_once():
    ids = np.repeat(np.arange(11).astype(str), np.arange(1, 12))
    coverage = np.zeros(len(ids), dtype=int)
    for train, hold in rd.image_folds(ids):
        rd.assert_group_disjoint(ids[train], ids[hold])
        assert set(train) | set(hold) == set(range(len(ids)))
        coverage[hold] += 1
    assert np.all(coverage == 1)
    with pytest.raises(ValueError, match="overlap"):
        rd.assert_group_disjoint(np.array(["a", "b"]), np.array(["b", "c"]))


def test_multimodal_feature_definition_is_exact():
    data = synthetic_dataset()
    combined = data.features("multimodal")
    expected = np.concatenate((data.image_features, data.text_features,
                               data.image_features * data.text_features,
                               np.abs(data.image_features - data.text_features)), axis=1)
    np.testing.assert_array_equal(combined, expected)
    assert data.features("image").shape[1] == 3
    assert data.features("text").shape[1] == 3


def test_standardization_fits_training_data_only_and_selection_is_validation_only():
    x_train = np.tile(np.eye(3), (4, 1))
    y_train = np.tile(np.arange(3), 4)
    x_validation = np.eye(3) + 1000
    x = np.vstack((x_train, x_validation))
    y = np.r_[y_train, np.arange(3)]
    groups = np.array([f"fit:{i}" for i in range(12)] + [f"val:{i}" for i in range(3)])
    model, record = rd.fit_select(x, y, np.arange(12), np.arange(12, 15), groups, c_grid=(.1, 1.))
    np.testing.assert_allclose(model.named_steps["scale"].mean_, x_train.mean(axis=0))
    assert record["selected_C"] == .1  # Same perfect ordering, lower C tie break.
    assert all(item["validation_macro_f1"] == 1 for item in record["candidates"])


def test_temperature_fit_uses_scalar_nll_and_never_worsens_its_fit_objective():
    logits = np.array([[10., 0., 0.], [10., 0., 0.], [0., 10., 0.]])
    labels = np.array([0, 1, 1])
    result = rd.fit_temperature(logits, labels)
    assert .05 <= result["temperature"] <= 20
    assert result["temperature"] > 1
    assert result["fit_nll_after"] <= result["fit_nll_before"]
    assert np.array_equal(logits.argmax(axis=1), (logits / result["temperature"]).argmax(axis=1))


def test_wilson_gate_requires_enough_evidence_even_for_perfect_predictions():
    assert rd.wilson_lower(34, 34) < .9 < rd.wilson_lower(35, 35)
    probs = np.tile([.95, .03, .02], (34, 1))
    rejected = rd.select_gate(probs, np.zeros(34, dtype=int), 0)
    assert rejected["threshold"] is None
    probs = np.tile([.95, .03, .02], (35, 1))
    accepted = rd.select_gate(probs, np.zeros(35, dtype=int), 0)
    assert accepted["selected_count"] == 35 and accepted["selected_wilson_lower"] >= .9


def test_gate_maximizes_coverage_and_does_not_use_probability_of_other_predictions():
    probs = np.vstack((np.tile([.95, .03, .02], (90, 1)), np.tile([.8, .1, .1], (15, 1)),
                       np.tile([.49, .50, .01], (20, 1))))
    labels = np.r_[np.zeros(90, dtype=int), np.ones(15, dtype=int), np.zeros(20, dtype=int)]
    gate = rd.select_gate(probs, labels, 0)
    assert gate["threshold"] == .95 and gate["selected_count"] == 90
    assert gate["predicted_class_count"] == 105
    assert np.sum(rd.acceptance_mask(probs, gate)) == 90
    assert gate["selected_coverage"] == 90 / 125


def test_threshold_selection_cannot_break_confidence_ties_to_hide_errors():
    probs = np.tile([.95, .03, .02], (60, 1))
    labels = np.r_[np.zeros(35, dtype=int), np.ones(25, dtype=int)]
    assert rd.select_gate(probs, labels, 0)["threshold"] is None


def test_gate_coverage_and_precision_use_different_denominators():
    probs = np.vstack((np.tile([.95, .03, .02], (40, 1)), np.tile([.1, .8, .1], (60, 1))))
    labels = np.r_[np.zeros(35, dtype=int), np.ones(65, dtype=int)]
    gate = {"class_index": 0, "threshold": .9}
    report = rd.gate_report(probs, labels, np.repeat(np.arange(10).astype(str), 10), gate, bootstrap_replicates=100)
    assert report["coverage"] == .4 and report["precision"] == 35 / 40
    empty = rd.gate_report(probs, labels, np.repeat(np.arange(10).astype(str), 10), {"class_index": 0, "threshold": None}, 100)
    assert empty["accepted"] == 0 and empty["precision"] is None
    assert empty["precision_image_bootstrap"]["interval"] is None


def test_metrics_match_perfect_and_uniform_probability_calculations():
    labels = np.arange(3)
    perfect = rd.calibration_metrics(np.eye(3), labels)
    assert perfect["accuracy"] == 1 and perfect["macro_f1"] == 1
    assert perfect["ece_10_bins"] == 0 and perfect["brier_sum_classes"] == 0 and perfect["nll"] == 0
    uniform = rd.calibration_metrics(np.full((3, 3), 1 / 3), labels)
    assert uniform["accuracy"] == 1 / 3
    assert uniform["ece_10_bins"] == 0
    assert uniform["brier_sum_classes"] == pytest.approx(2 / 3)
    assert uniform["nll"] == pytest.approx(np.log(3))
    assert sum(item["count"] for item in perfect["reliability_bins"]) == 3
    assert perfect["reliability_bins"][-1]["count"] == 3


def test_precision_bootstrap_resamples_image_clusters_not_individual_pairs():
    images = np.array(["a"] * 100 + ["b"])
    labels = np.r_[np.zeros(100, dtype=int), 1]
    accepted = np.ones(101, dtype=bool)
    result = rd.precision_bootstrap(accepted, labels, images, 0, replicates=1000, seed=19)
    assert result["cluster_count"] == 2 and result["valid_replicates"] == 1000
    assert result["interval"] == [0., 1.]
    assert result == rd.precision_bootstrap(accepted, labels, images, 0, replicates=1000, seed=19)


def test_crossfit_predictions_are_invariant_to_held_out_image_labels():
    data = synthetic_dataset()
    x = data.features("text")
    training, validation, calibration = (data.rows(split) for split in ("train", "validation", "calibration"))
    fit_local, hold_local = rd.image_folds(data.image_ids[training])[0]
    fit_rows, hold_rows = training[fit_local], training[hold_local]
    temp_local, gate_local = rd.split_calibration_images(data.image_ids[calibration])
    temp_rows, gate_rows = calibration[temp_local], calibration[gate_local]
    first, first_selection, first_temperature, first_gates = rd.fit_calibrated(x, data, fit_rows, validation, temp_rows, gate_rows, c_grid=(.1, 1.))
    expected = softmax(first.decision_function(x[hold_rows]) / first_temperature["temperature"], axis=1)
    # This emulates labels that must be unavailable to the fold's whole fit path.
    data.labels[hold_rows] = (data.labels[hold_rows] + 1) % 3
    second, second_selection, second_temperature, second_gates = rd.fit_calibrated(x, data, fit_rows, validation, temp_rows, gate_rows, c_grid=(.1, 1.))
    actual = softmax(second.decision_function(x[hold_rows]) / second_temperature["temperature"], axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert first_selection == second_selection and first_temperature == second_temperature and first_gates == second_gates


def test_saved_numeric_model_reconstructs_predictions_without_pickle(tmp_path):
    data = synthetic_dataset()
    x = data.features("text")
    train, validation = data.rows("train"), data.rows("validation")
    model, _ = rd.fit_select(x, data.labels, train, validation, data.image_ids, c_grid=(.1,))
    path = tmp_path / "classifier.npz"
    rd.save_model(path, model, {"temperature": 1.2})
    with np.load(path, allow_pickle=False) as saved:
        scores = ((x[validation] - saved["scaler_mean"]) / saved["scaler_scale"]) @ saved["coefficients"].T + saved["intercept"]
        np.testing.assert_allclose(scores, model.decision_function(x[validation]), atol=1e-14)
        assert saved["temperature"].item() == 1.2


def test_feature_loader_excludes_source_pairs_and_binds_manifest_hash(tmp_path):
    images, texts, pairs, image_features, text_features = [], [], [], [], []
    for split in ("train", "validation", "calibration", "test"):
        images.append({"id": split, "split": split, "path": "unused.jpg"})
        image_features.append([1., 0., 0.])
        for i, relation in enumerate(("source", "supported", "contradicted", "neutral")):
            text_id = f"{split}:{relation}"
            texts.append({"id": text_id, "text": text_id})
            text_features.append(np.eye(3)[i % 3])
            pairs.append({"image_id": split, "text_id": text_id, "relation": relation})
    manifest = tmp_path / "manifest.json"
    features = tmp_path / "features.npz"
    metadata = tmp_path / "metadata.json"
    manifest.write_text(json.dumps({"images": images, "texts": texts, "pairs": pairs}))
    np.savez(features, image_features=image_features, text_features=text_features,
             image_ids=np.array([item["id"] for item in images]), text_ids=np.array([item["id"] for item in texts]))
    metadata.write_text(json.dumps({"manifest_sha256": rd.sha256(manifest)}))
    loaded = rd.load_relation_dataset(manifest, features, metadata)
    assert len(loaded.labels) == 12 and np.bincount(loaded.labels).tolist() == [4, 4, 4]
    assert not any(":source" in text for text in loaded.text_ids)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="bind"):
        rd.load_relation_dataset(manifest, features, metadata)


def test_small_complete_execution_saves_reusable_auditable_outputs(tmp_path):
    from types import SimpleNamespace
    data = synthetic_dataset()
    image_ids = list(dict.fromkeys(data.image_ids))
    first_rows = [np.flatnonzero(data.image_ids == image)[0] for image in image_ids]
    manifest = tmp_path / "manifest.json"
    features = tmp_path / "features.npz"
    metadata = tmp_path / "metadata.json"
    images = [{"id": image, "split": str(data.splits[row]), "path": "unused.jpg"} for image, row in zip(image_ids, first_rows)]
    texts = [{"id": str(text), "text": str(text)} for text in data.text_ids]
    pairs = [{"image_id": str(image), "text_id": str(text), "relation": rd.CLASSES[label]}
             for image, text, label in zip(data.image_ids, data.text_ids, data.labels)]
    manifest.write_text(json.dumps({"images": images, "texts": texts, "pairs": pairs}))
    np.savez(features, image_features=data.image_features[first_rows], text_features=data.text_features,
             image_ids=np.array(image_ids), text_ids=data.text_ids)
    metadata.write_text(json.dumps({"manifest_sha256": rd.sha256(manifest), "features_sha256": rd.sha256(features)}))
    args = SimpleNamespace(manifest=manifest, features=features, metadata=metadata, output=tmp_path / "output", threads=1)
    rd.run(args)
    first_summary_hash = rd.sha256(args.output / "summary.json")
    for mode in rd.FEATURE_MODES:
        with np.load(args.output / mode / "train_out_of_fold_predictions.npz", allow_pickle=False) as prediction:
            assert len(prediction["labels"]) == len(data.rows("train"))
            assert set(prediction["fold_indices"]) == {0, 1, 2}
            np.testing.assert_allclose(prediction["probabilities"].sum(1), 1)
        assert (args.output / mode / "FILE_HASHES.json").exists()
    rd.run(args)
    assert rd.sha256(args.output / "summary.json") == first_summary_hash
    # A corrupted completed checkpoint must not be silently reused.
    (args.output / "image" / "classifier.npz").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="missing or changed"):
        rd.run(args)
