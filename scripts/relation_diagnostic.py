#!/usr/bin/env python3
"""Appendix-only, image-disjoint three-way relation-estimator diagnostic.

No output from this script trains or selects a primary contrastive adapter.
See docs/RELATION_DIAGNOSTIC_PROTOCOL.md for the reconstruction decisions.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("supported", "contradicted", "neutral")
FEATURE_MODES = ("image", "text", "multimodal")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


@dataclasses.dataclass
class RelationDataset:
    image_features: np.ndarray
    text_features: np.ndarray
    labels: np.ndarray
    image_ids: np.ndarray
    text_ids: np.ndarray
    splits: np.ndarray
    source_pair_indices: np.ndarray

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)

    def features(self, mode: str) -> np.ndarray:
        if mode == "image":
            return np.asarray(self.image_features, dtype=np.float64)
        if mode == "text":
            return np.asarray(self.text_features, dtype=np.float64)
        if mode == "multimodal":
            image, text = self.image_features, self.text_features
            return np.concatenate((image, text, image * text, np.abs(image - text)), axis=1).astype(np.float64)
        raise ValueError(f"Unknown feature mode: {mode}")


def load_relation_dataset(manifest_path: Path, features_path: Path, metadata_path: Path) -> RelationDataset:
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    declared = metadata.get("manifest_sha256", metadata.get("input_manifest_sha256"))
    if declared != sha256(manifest_path):
        raise ValueError("Feature metadata does not bind the supplied manifest")
    if metadata.get("features_sha256") and metadata["features_sha256"] != sha256(features_path):
        raise ValueError("Feature content hash does not match metadata")
    with np.load(features_path, allow_pickle=False) as cache:
        images = np.array(cache["image_features"], dtype=np.float32)
        texts = np.array(cache["text_features"], dtype=np.float32)
        image_ids = cache["image_ids"].astype(str)
        text_ids = cache["text_ids"].astype(str)
    if image_ids.tolist() != [str(x["id"]) for x in manifest["images"]] or text_ids.tolist() != [str(x["id"]) for x in manifest["texts"]]:
        raise ValueError("Feature rows must exactly match manifest order")
    if len(set(image_ids)) != len(image_ids) or len(set(text_ids)) != len(text_ids):
        raise ValueError("Manifest image and text IDs must be unique")
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1] or images.shape[0] != len(image_ids) or texts.shape[0] != len(text_ids):
        raise ValueError("Invalid feature matrix dimensions")
    for features in (images, texts):
        if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=2e-4, rtol=2e-4):
            raise ValueError("Features must be finite and L2 normalized")
    image_lookup = {image: i for i, image in enumerate(image_ids)}
    text_lookup = {text: i for i, text in enumerate(text_ids)}
    label_lookup = {name: i for i, name in enumerate(CLASSES)}
    row_images, row_texts, labels, splits, source_indices = [], [], [], [], []
    seen = set()
    text_splits = {}
    for pair_index, pair in enumerate(manifest["pairs"]):
        relation = pair["relation"]
        if relation == "source":
            continue
        if relation not in label_lookup:
            raise ValueError(f"Unknown hypothesis relation: {relation}")
        image, text = image_lookup[pair["image_id"]], text_lookup[pair["text_id"]]
        identity = (image, text)
        if identity in seen:
            raise ValueError("Duplicate hypothesis pair in manifest")
        seen.add(identity)
        split = manifest["images"][image]["split"]
        if split not in ("train", "validation", "calibration", "test"):
            raise ValueError(f"Unknown split: {split}")
        text_splits.setdefault(text, set()).add(split)
        row_images.append(image)
        row_texts.append(text)
        labels.append(label_lookup[relation])
        splits.append(split)
        source_indices.append(pair_index)
    if any(len(value) > 1 for value in text_splits.values()):
        raise ValueError("A hypothesis ID is shared across data splits")
    if not row_images:
        raise ValueError("No hypothesis annotations found")
    row_images, row_texts = np.array(row_images), np.array(row_texts)
    result = RelationDataset(images[row_images], texts[row_texts], np.asarray(labels, dtype=np.int64),
                             image_ids[row_images], text_ids[row_texts], np.asarray(splits), np.asarray(source_indices))
    for split in ("train", "validation", "calibration", "test"):
        if not result.rows(split).size:
            raise ValueError(f"Missing nonempty {split} hypothesis split")
    assert_group_disjoint(*(result.image_ids[result.rows(split)] for split in ("train", "validation", "calibration", "test")))
    return result


def assert_group_disjoint(*groups: np.ndarray) -> None:
    sets = [set(np.asarray(values).astype(str)) for values in groups]
    for i, left in enumerate(sets):
        for right in sets[i + 1:]:
            if left & right:
                raise ValueError("Image groups overlap between fitting and held-out partitions")


def split_calibration_images(image_ids: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(image_ids)
    if unique.size < 2:
        raise ValueError("At least two calibration image groups are required")
    permuted = np.random.default_rng(seed).permutation(unique)
    middle = len(permuted) // 2
    temperature = np.flatnonzero(np.isin(image_ids, permuted[:middle]))
    gate = np.flatnonzero(np.isin(image_ids, permuted[middle:]))
    assert_group_disjoint(image_ids[temperature], image_ids[gate])
    return temperature, gate


def image_folds(image_ids: np.ndarray, n_folds: int = 3, seed: int = 17) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(image_ids)
    if n_folds < 2 or len(unique) < n_folds:
        raise ValueError("Insufficient image groups for cross-fitting")
    groups = np.array_split(np.random.default_rng(seed).permutation(unique), n_folds)
    result = []
    for held_out in groups:
        test = np.flatnonzero(np.isin(image_ids, held_out))
        train = np.flatnonzero(~np.isin(image_ids, held_out))
        assert_group_disjoint(image_ids[train], image_ids[test])
        result.append((train, test))
    return result


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, c: float, seed: int = 17) -> Pipeline:
    if set(np.unique(y_train)) != set(range(3)):
        raise ValueError("Each classifier fitting partition must contain all three classes")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=c, solver="lbfgs", max_iter=10000, tol=1e-6,
                                          random_state=seed, class_weight=None)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    failures = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if failures:
        raise RuntimeError("Logistic regression did not converge: " + "; ".join(failures))
    return model


def fit_select(x: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray,
               image_ids: np.ndarray, c_grid=C_GRID, seed: int = 17) -> tuple[Pipeline, dict]:
    assert_group_disjoint(image_ids[train], image_ids[validation])
    candidates = []
    best_model, best_score, best_c = None, -np.inf, None
    for c in sorted(c_grid):
        print(f"  logistic C={float(c):g}; training rows={len(train)}", flush=True)
        model = fit_logistic(x[train], y[train], float(c), seed)
        prediction = model.predict(x[validation])
        score = float(f1_score(y[validation], prediction, labels=np.arange(3), average="macro", zero_division=0))
        candidates.append({"C": float(c), "validation_macro_f1": score,
                           "iterations": model.named_steps["classifier"].n_iter_.tolist()})
        if score > best_score:
            best_model, best_score, best_c = model, score, float(c)
    return best_model, {"selected_C": best_c, "validation_macro_f1": best_score, "candidates": candidates,
                        "training_image_ids": np.unique(image_ids[train]).tolist(),
                        "validation_image_ids": np.unique(image_ids[validation]).tolist()}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    if logits.ndim != 2 or logits.shape[1] != 3 or len(labels) != len(logits) or not len(labels):
        raise ValueError("Temperature fitting requires nonempty three-class logits and labels")
    if not np.isfinite(logits).all() or not np.isin(labels, np.arange(3)).all():
        raise ValueError("Invalid logits or class labels")
    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))
    solution = minimize_scalar(objective, bounds=(np.log(.05), np.log(20.)), method="bounded",
                               options={"xatol": 1e-8, "maxiter": 1000})
    if not solution.success:
        raise RuntimeError("Temperature optimization failed")
    candidates = [0., float(solution.x), float(np.log(.05)), float(np.log(20.))]
    chosen = min(candidates, key=objective)
    return {"temperature": float(np.exp(chosen)), "fit_nll_before": objective(0.),
            "fit_nll_after": objective(chosen), "bounds": [.05, 20.], "examples": len(labels)}


def wilson_lower(correct: int | np.ndarray, count: int | np.ndarray, z: float = 1.959963984540054) -> float | np.ndarray:
    correct, count = np.asarray(correct, dtype=float), np.asarray(count, dtype=float)
    if np.any(count <= 0) or np.any(correct < 0) or np.any(correct > count):
        raise ValueError("Wilson counts must satisfy 0 <= correct <= count and count > 0")
    p = correct / count
    result = (p + z * z / (2 * count) - z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count))) / (1 + z * z / count)
    return float(result) if result.ndim == 0 else result


def select_gate(probabilities: np.ndarray, labels: np.ndarray, class_index: int,
                min_count: int = 20, min_wilson_lower: float = .90) -> dict:
    prediction = np.argmax(probabilities, axis=1)
    eligible = np.flatnonzero(prediction == class_index)
    base = {"class": CLASSES[class_index], "class_index": class_index, "min_count": min_count,
            "min_wilson_lower": min_wilson_lower, "threshold": None, "selected_count": 0,
            "selected_correct": 0, "selected_precision": None, "selected_wilson_lower": None,
            "selected_coverage": 0., "selection_examples": len(labels), "predicted_class_count": len(eligible)}
    if len(eligible) < min_count:
        return base
    # Stable descending order and whole tie groups ensure >= threshold semantics.
    order = eligible[np.argsort(-probabilities[eligible, class_index], kind="stable")]
    confidence = probabilities[order, class_index]
    correct = np.cumsum(labels[order] == class_index)
    ends = np.r_[np.flatnonzero(confidence[:-1] != confidence[1:]), len(order) - 1]
    counts = ends + 1
    lowers = wilson_lower(correct[ends], counts)
    feasible = np.flatnonzero((counts >= min_count) & (lowers >= min_wilson_lower))
    if not len(feasible):
        return base
    position = feasible[-1]  # Maximum count, hence maximum coverage.
    end, count = int(ends[position]), int(counts[position])
    base.update(threshold=float(confidence[end]), selected_count=count, selected_correct=int(correct[end]),
                selected_precision=float(correct[end] / count), selected_wilson_lower=float(lowers[position]),
                selected_coverage=float(count / len(labels)))
    return base


def acceptance_mask(probabilities: np.ndarray, gate: dict) -> np.ndarray:
    if gate["threshold"] is None:
        return np.zeros(len(probabilities), dtype=bool)
    class_index = gate["class_index"]
    return (probabilities.argmax(axis=1) == class_index) & (probabilities[:, class_index] >= gate["threshold"])


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    if probabilities.shape != (len(labels), 3) or not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Metrics require nonempty finite three-class probabilities")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Rows must contain probability distributions")
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == labels
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.
    for index in range(bins):
        mask = bin_ids == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    target = np.eye(3)[labels]
    return {"examples": len(labels), "accuracy": float(accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, labels=np.arange(3), average="macro", zero_division=0)),
            "ece_10_bins": float(ece), "brier_sum_classes": float(np.mean(np.sum((probabilities - target) ** 2, axis=1))),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-300, 1)).mean()),
            "reliability_bins": reliability}


def precision_bootstrap(accepted: np.ndarray, labels: np.ndarray, image_ids: np.ndarray, class_index: int,
                        replicates: int = 10000, seed: int = 20261003) -> dict:
    unique, inverse = np.unique(image_ids, return_inverse=True)
    counts = np.bincount(inverse, weights=accepted.astype(int), minlength=len(unique))
    correct = np.bincount(inverse, weights=(accepted & (labels == class_index)).astype(int), minlength=len(unique))
    base = {"replicates": replicates, "seed": seed, "level": .95, "resampling_unit": "image",
            "cluster_count": len(unique), "interval": None, "valid_replicates": 0, "empty_replicates": 0,
            "interpretation": "descriptive percentile precision interval conditional on the selected gate"}
    if counts.sum() == 0:
        return base
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, replicates, 512):
        draw = rng.integers(0, len(unique), size=(min(512, replicates - start), len(unique)))
        denominator = counts[draw].sum(axis=1)
        numerator = correct[draw].sum(axis=1)
        valid = denominator > 0
        values.extend((numerator[valid] / denominator[valid]).tolist())
    base.update(valid_replicates=len(values), empty_replicates=replicates - len(values))
    if values:
        base["interval"] = np.quantile(values, [.025, .975]).tolist()
    return base


def gate_report(probabilities: np.ndarray, labels: np.ndarray, image_ids: np.ndarray, gate: dict,
                bootstrap_replicates: int = 10000) -> dict:
    accepted = acceptance_mask(probabilities, gate)
    count = int(accepted.sum())
    correct = int(np.sum(labels[accepted] == gate["class_index"]))
    return {"threshold": gate["threshold"], "accepted": count, "correct": correct,
            "coverage": float(count / len(labels)), "precision": correct / count if count else None,
            "precision_image_bootstrap": precision_bootstrap(accepted, labels, image_ids, gate["class_index"], bootstrap_replicates)}


def save_model(path: Path, model: Pipeline, temperature: dict) -> None:
    scale = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    atomic_npz(path, scaler_mean=scale.mean_, scaler_scale=scale.scale_, scaler_variance=scale.var_,
               coefficients=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_,
               temperature=np.asarray(temperature["temperature"]))


def save_predictions(path: Path, data: RelationDataset, rows: np.ndarray, logits: np.ndarray,
                     probabilities: np.ndarray, gates: dict, fold_indices: np.ndarray | None = None) -> None:
    arrays = dict(image_ids=data.image_ids[rows], text_ids=data.text_ids[rows], labels=data.labels[rows],
                  manifest_pair_indices=data.source_pair_indices[rows], logits=logits, probabilities=probabilities,
                  prediction=probabilities.argmax(axis=1),
                  accepted_supported=acceptance_mask(probabilities, gates["supported"]),
                  accepted_contradicted=acceptance_mask(probabilities, gates["contradicted"]))
    if fold_indices is not None:
        arrays["fold_indices"] = fold_indices
    atomic_npz(path, **arrays)


def fit_calibrated(x: np.ndarray, data: RelationDataset, train: np.ndarray, validation: np.ndarray,
                   temperature_rows: np.ndarray, gate_rows: np.ndarray, c_grid=C_GRID) -> tuple[Pipeline, dict, dict, dict]:
    assert_group_disjoint(*(data.image_ids[rows] for rows in (train, validation, temperature_rows, gate_rows)))
    model, selection = fit_select(x, data.labels, train, validation, data.image_ids, c_grid)
    temperature = fit_temperature(model.decision_function(x[temperature_rows]), data.labels[temperature_rows])
    probabilities = softmax(model.decision_function(x[gate_rows]) / temperature["temperature"], axis=1)
    gates = {CLASSES[index]: select_gate(probabilities, data.labels[gate_rows], index) for index in (0, 1)}
    return model, selection, temperature, gates


def protocol() -> dict:
    return {"schema_version": 1, "scope": "appendix-only relation diagnostic; never used for adapter training",
            "classes": list(CLASSES), "feature_modes": list(FEATURE_MODES), "C_grid": list(C_GRID), "classifier_seed": 17,
            "solver": "lbfgs", "standardization": "fit on classifier training rows only", "class_weight": None,
            "max_iter": 10000, "tol": 1e-6, "selection": "validation macro-F1; lower C breaks exact ties",
            "calibration_split_seed": 0, "calibration_split": "equal image halves: temperature then gates",
            "temperature_bounds": [.05, 20.], "gate_min_count": 20, "gate_wilson_lower": .90,
            "gate_wilson_confidence": .95, "gate_selection": "maximum accepted count over whole confidence-tie groups",
            "crossfit_folds": 3, "crossfit_seed": 17, "crossfit_selection": "reselect C and recalibrate separately within every fold",
            "bootstrap_replicates": 10000, "bootstrap_seed": 20261003, "bootstrap_unit": "image",
            "brier_definition": "mean sum over three class squared errors", "ece_bins": 10,
            "reconstruction": "newly specified diagnostic hyperparameters; original implementation was unavailable"}


def run(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {"protocol": protocol(), "source_sha256": sha256(Path(__file__)),
              "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in (
                  ("manifest", args.manifest), ("features", args.features), ("metadata", args.metadata))},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "threads": args.threads}}
    ledger_path = output / "protocol_ledger.json"
    if ledger_path.exists() and json.loads(ledger_path.read_text()) != record:
        raise RuntimeError("Existing diagnostic ledger differs; use a new output directory")
    atomic_json(ledger_path, record)
    data = load_relation_dataset(args.manifest, args.features, args.metadata)
    train, validation, calibration, test = (data.rows(split) for split in ("train", "validation", "calibration", "test"))
    temp_local, gate_local = split_calibration_images(data.image_ids[calibration])
    temperature_rows, gate_rows = calibration[temp_local], calibration[gate_local]
    partitions = {name: {"rows": len(rows), "image_ids": np.unique(data.image_ids[rows]).tolist(),
                         "class_counts": np.bincount(data.labels[rows], minlength=3).tolist()} for name, rows in (
                             ("train", train), ("validation", validation), ("temperature", temperature_rows),
                             ("gate_selection", gate_rows), ("test", test))}
    atomic_json(output / "partitions.json", partitions)
    summary = {"scope": record["protocol"]["scope"], "classes": CLASSES, "modes": {}}
    for mode in FEATURE_MODES:
        start = time.monotonic()
        destination = output / mode
        complete = destination / "summary.json"
        if complete.exists():
            receipt_path = destination / "FILE_HASHES.json"
            if not receipt_path.exists():
                raise RuntimeError(f"Completed mode {mode} lacks its integrity receipt")
            receipt = json.loads(receipt_path.read_text())
            for name, expected in receipt.items():
                path = destination / name
                if not path.is_file() or sha256(path) != expected["sha256"]:
                    raise RuntimeError(f"Completed mode {mode} has a missing or changed file: {name}")
            summary["modes"][mode] = json.loads(complete.read_text())
            print(f"REUSE completed diagnostic mode {mode}", flush=True)
            continue
        x = data.features(mode)
        print(f"FIT relation diagnostic {mode}: {x.shape[1]} features", flush=True)
        model, selection, temperature, gates = fit_calibrated(x, data, train, validation, temperature_rows, gate_rows)
        save_model(destination / "classifier.npz", model, temperature)
        outputs = {"selection": selection, "temperature": temperature, "gates": gates, "metrics": {}, "crossfit": []}
        for split, rows in (("validation", validation), ("temperature_fit", temperature_rows), ("gate_selection", gate_rows), ("test", test)):
            logits = model.decision_function(x[rows])
            raw, probabilities = softmax(logits, axis=1), softmax(logits / temperature["temperature"], axis=1)
            outputs["metrics"][split] = {"uncalibrated": calibration_metrics(raw, data.labels[rows]),
                                          "calibrated": calibration_metrics(probabilities, data.labels[rows])}
            save_predictions(destination / f"{split}_predictions.npz", data, rows, logits, probabilities, gates)
            if split == "test":
                outputs["test_gates"] = {name: gate_report(probabilities, data.labels[rows], data.image_ids[rows], gate) for name, gate in gates.items()}
        # Every row is predicted by a model that excludes its entire image group.
        oof_logits = np.empty((len(train), 3))
        oof_probabilities = np.empty_like(oof_logits)
        oof_folds = np.full(len(train), -1, dtype=np.int64)
        oof_acceptance = {name: np.zeros(len(train), dtype=bool) for name in ("supported", "contradicted")}
        for fold, (fit_local, hold_local) in enumerate(image_folds(data.image_ids[train])):
            fit_rows, hold_rows = train[fit_local], train[hold_local]
            assert_group_disjoint(data.image_ids[fit_rows], data.image_ids[hold_rows])
            fold_model, fold_selection, fold_temperature, fold_gates = fit_calibrated(x, data, fit_rows, validation, temperature_rows, gate_rows)
            logits = fold_model.decision_function(x[hold_rows])
            probabilities = softmax(logits / fold_temperature["temperature"], axis=1)
            oof_logits[hold_local], oof_probabilities[hold_local], oof_folds[hold_local] = logits, probabilities, fold
            for name, gate in fold_gates.items():
                oof_acceptance[name][hold_local] = acceptance_mask(probabilities, gate)
            save_model(destination / f"crossfit_{fold}_classifier.npz", fold_model, fold_temperature)
            save_predictions(destination / f"crossfit_{fold}_predictions.npz", data, hold_rows, logits, probabilities, fold_gates)
            outputs["crossfit"].append({"fold": fold, "selection": fold_selection, "temperature": fold_temperature,
                                        "gates": fold_gates, "held_out_image_ids": np.unique(data.image_ids[hold_rows]).tolist()})
            print(f"COMPLETE {mode} crossfit fold {fold}", flush=True)
        if np.any(oof_folds < 0):
            raise RuntimeError("Cross-fitting did not assign all training hypothesis rows")
        atomic_npz(destination / "train_out_of_fold_predictions.npz", image_ids=data.image_ids[train], text_ids=data.text_ids[train],
                   labels=data.labels[train], manifest_pair_indices=data.source_pair_indices[train], logits=oof_logits,
                   probabilities=oof_probabilities, prediction=oof_probabilities.argmax(axis=1), fold_indices=oof_folds,
                   accepted_supported=oof_acceptance["supported"], accepted_contradicted=oof_acceptance["contradicted"])
        outputs["out_of_fold_metrics"] = calibration_metrics(oof_probabilities, data.labels[train])
        outputs["elapsed_seconds"] = time.monotonic() - start
        atomic_json(complete, outputs)
        mode_hashes = {str(path.relative_to(destination)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                       for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "FILE_HASHES.json"}
        atomic_json(destination / "FILE_HASHES.json", mode_hashes)
        summary["modes"][mode] = outputs
        atomic_json(output / "summary.json", summary)
        print(f"COMPLETE relation diagnostic {mode}; test macro-F1={outputs['metrics']['test']['calibrated']['macro_f1']:.6f}", flush=True)
    atomic_json(output / "summary.json", summary)
    files = {str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file() and path.name != "FILE_HASHES.json"}
    atomic_json(output / "FILE_HASHES.json", files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/visual_entailment/manifest.json")
    parser.add_argument("--features", type=Path, default=ROOT / "results/features/visual_entailment/features.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT / "results/features/visual_entailment/metadata.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/relation_diagnostic")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
