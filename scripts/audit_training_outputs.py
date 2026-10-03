#!/usr/bin/env python3
"""Independently verify fitted artifacts and selection without held-out scores."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/study")
    parser.add_argument("--output", type=Path, default=ROOT / "results/training_final_audit.json")
    args = parser.parse_args()
    failures, checks = [], []

    def check(name, passed, **evidence):
        record = {"check": name, "passed": bool(passed), **evidence}
        checks.append(record)
        if not passed:
            failures.append(record)

    ledger = read(args.study / "ledger.json")
    identity = ledger["identity"]
    expected_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    check("ledger_identity_digest", expected_hash == ledger["ledger_sha256"], sha256=expected_hash)
    protocol_path = ROOT / identity["protocol"]["path"]
    check("frozen_protocol_unchanged", digest(protocol_path) == identity["protocol"]["sha256"], sha256=digest(protocol_path))
    for path, expected in identity["source_sha256"].items():
        check("frozen_source_unchanged", digest(ROOT / path) == expected, path=path, sha256=digest(ROOT / path))
    for kind in ("manifest", "features", "metadata"):
        entry = identity["inputs"][kind]
        actual = digest(ROOT / entry["path"])
        check("input_unchanged", actual == entry["sha256"], kind=kind, sha256=actual)
    hp = identity["hyperparameters"]
    protocol = read(protocol_path)
    check("declared_full_grid", hp["methods"] == protocol["training"]["methods"] and hp["learning_rates"] == protocol["training"]["learning_rates"] and hp["seeds"] == protocol["training"]["seeds"] and hp["epochs"] == 10,
          methods=hp["methods"], learning_rates=hp["learning_rates"], seeds=hp["seeds"], epochs=hp["epochs"])
    manifest = read(ROOT / identity["inputs"]["manifest"]["path"])
    split_ids = {}
    for row in manifest["images"]:
        split_ids.setdefault(row["split"], set()).add(row["id"])
    split_overlap = {f"{a}/{b}": len(split_ids[a] & split_ids[b]) for a, b in itertools.combinations(sorted(split_ids), 2)}
    check("image_split_disjointness", not any(split_overlap.values()), overlaps=split_overlap,
          counts={key: len(value) for key, value in split_ids.items()})
    check("ledger_train_and_validation_ids", set(identity["inputs"]["training_image_ids"]) == split_ids["train"] and set(identity["inputs"]["validation_image_ids"]) == split_ids["validation"])
    validation_ids = identity["inputs"]["validation_image_ids"]
    validation_text_ids = set(identity["inputs"]["validation_text_ids"])
    baseline_validation = read(args.study / "epoch_zero_validation.json")
    baseline_checkpoint = torch.load(args.study / "epoch_zero.pt", map_location="cpu", weights_only=True)
    check("epoch_zero_is_zero_residual", baseline_checkpoint["epoch"] == 0 and baseline_checkpoint["ledger_sha256"] == ledger["ledger_sha256"] and all(torch.count_nonzero(value).item() == 0 for value in baseline_checkpoint["state_dict"].values()))
    check("no_heldout_scores_in_selection_rule", identity["historical_results_read"] is False and identity["selection"]["held_out_data_used"] is False)
    candidates = {}
    history_epochs = 0
    max_metric_error = 0.0
    artifact_errors, history_errors, identity_errors, best_errors = [], [], [], []
    expected_grid = set(itertools.product(hp["methods"], hp["learning_rates"], hp["seeds"]))
    observed_grid = set()
    for receipt_path in sorted((args.study / "candidates").glob("*/lr_*/seed_*/completion.json")):
        receipt = read(receipt_path)
        key = (receipt["method"], receipt["learning_rate"], receipt["seed"])
        if key in observed_grid:
            identity_errors.append({"duplicate": list(key)})
        observed_grid.add(key)
        directory = receipt_path.parent
        for artifact in ("best.pt", "history.json"):
            if digest(directory / artifact) != receipt["artifacts_sha256"][artifact]:
                artifact_errors.append(str(directory / artifact))
        history = read(directory / "history.json")
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        for payload in (receipt, history, checkpoint):
            if (payload.get("method"), payload.get("learning_rate"), payload.get("seed")) != key or payload.get("ledger_sha256") != ledger["ledger_sha256"]:
                identity_errors.append({"candidate": list(key), "reason": "identity mismatch"})
        if receipt["epochs_completed"] != hp["epochs"] or [row["epoch"] for row in history["history"]] != list(range(hp["epochs"] + 1)):
            history_errors.append({"candidate": list(key), "reason": "epoch range"})
        for row in history["history"]:
            history_epochs += 1
            validation = row["validation"]
            rows = validation["per_image"]
            r1 = float(np.mean([x["known_positive_i2t_r1"] for x in rows]))
            relations = [x["relation_accuracy"] for x in rows if x["relation_accuracy"] is not None]
            relation_accuracy = float(np.mean(relations))
            score = (r1 + relation_accuracy) / 2
            error = max(abs(r1 - validation["known_positive_i2t_r1"]), abs(relation_accuracy - validation["relation_accuracy"]), abs(score - validation["score"]))
            max_metric_error = max(max_metric_error, error)
            if error > 1e-14 or [x["image_id"] for x in rows] != validation_ids or any(x["winning_text_id"] not in validation_text_ids for x in rows):
                history_errors.append({"candidate": list(key), "epoch": row["epoch"], "reason": "raw metric mean or validation membership"})
            expected_steps = 0 if row["epoch"] == 0 else math.ceil(len(split_ids["train"]) / hp["image_batch_size"])
            expected_images = 0 if row["epoch"] == 0 else len(split_ids["train"])
            if row["gradient_steps"] != expected_steps or row["training_images"] != expected_images:
                history_errors.append({"candidate": list(key), "epoch": row["epoch"], "reason": "training step/image counts"})
            if row["epoch"] == 0 and validation != baseline_validation:
                history_errors.append({"candidate": list(key), "reason": "epoch-zero validation differs"})
        best = min(history["history"], key=lambda row: (-row["validation"]["score"], row["epoch"]))
        if receipt["best_epoch"] != best["epoch"] or history["best_epoch"] != best["epoch"] or checkpoint["epoch"] != best["epoch"] or checkpoint["validation"]["score"] != best["validation"]["score"] or receipt["best_validation_score"] != best["validation"]["score"]:
            best_errors.append({"candidate": list(key), "expected_epoch": best["epoch"]})
        candidates[key] = {"directory": directory, "best_epoch": best["epoch"], "best_score": best["validation"]["score"], "best_validation": best["validation"]}
    check("exact_108_candidate_grid", observed_grid == expected_grid and len(observed_grid) == 108, observed_count=len(observed_grid), missing=[list(x) for x in sorted(expected_grid - observed_grid)], extra=[list(x) for x in sorted(observed_grid - expected_grid)])
    check("all_candidate_receipt_artifact_hashes", not artifact_errors, failed_artifacts=artifact_errors)
    check("candidate_identity_bindings", not identity_errors, errors=identity_errors)
    check("all_raw_epoch_histories", not history_errors and history_epochs == 1188, checked_epochs=history_epochs, max_metric_aggregation_error=max_metric_error, errors=history_errors)
    check("best_epoch_including_tie_rule", not best_errors, errors=best_errors)
    selection = read(args.study / "selection.json")
    selection_errors, selected_rows = [], []
    for method in hp["methods"]:
        rate_scores = {rate: float(np.mean([candidates[(method, rate, seed)]["best_score"] for seed in hp["seeds"]])) for rate in hp["learning_rates"]}
        expected_rate = min(rate_scores, key=lambda rate: (-rate_scores[rate], rate))
        chosen = selection["methods"][method]
        if chosen["learning_rate"] != expected_rate or chosen["mean_validation_score"] != rate_scores[expected_rate] or {row["seed"] for row in chosen["runs"]} != set(hp["seeds"]):
            selection_errors.append({"method": method, "reason": "shared learning rate or seed set"})
        for run in chosen["runs"]:
            candidate = candidates[(method, expected_rate, run["seed"])]
            selected_digest = digest(ROOT / run["checkpoint"])
            candidate_digest = digest(candidate["directory"] / "best.pt")
            if selected_digest != run["checkpoint_sha256"] or selected_digest != candidate_digest or run["epoch"] != candidate["best_epoch"]:
                selection_errors.append({"method": method, "seed": run["seed"], "reason": "copied checkpoint or epoch"})
            selected_rows.append({"method": method, "learning_rate": expected_rate, **run, "expected_validation": candidate["best_validation"]})
    check("shared_rate_selection_and_checkpoint_copies", not selection_errors and len(selected_rows) == 36 and selection["ledger_sha256"] == ledger["ledger_sha256"] and selection["held_out_data_used"] is False,
          selected_count=len(selected_rows), errors=selection_errors)

    # Recompute selected-state validation independently of gcr.training and
    # gcr.evaluation. No held-out split features are passed to these scores.
    image_lookup = {row["id"]: index for index, row in enumerate(manifest["images"])}
    text_lookup = {row["id"]: index for index, row in enumerate(manifest["texts"])}
    image_indices = [image_lookup[iid] for iid in validation_ids]
    text_indices = sorted(text_lookup[tid] for tid in validation_text_ids)
    text_column = {manifest["texts"][index]["id"]: col for col, index in enumerate(text_indices)}
    image_row = {iid: row for row, iid in enumerate(validation_ids)}
    relations = torch.zeros(len(image_indices), len(text_indices), dtype=torch.int64)
    codes = {"source": 1, "supported": 2, "contradicted": 3, "neutral": 4}
    for pair in manifest["pairs"]:
        if pair["image_id"] in image_row:
            relations[image_row[pair["image_id"]], text_column[pair["text_id"]]] = codes[pair["relation"]]
    with np.load(ROOT / identity["inputs"]["features"]["path"], allow_pickle=False) as archive:
        images = torch.from_numpy(archive["image_features"][image_indices].copy())
        texts = torch.from_numpy(archive["text_features"][text_indices].copy())
    torch.set_num_threads(hp["threads"])
    state_errors, max_state_error = [], 0.0
    with torch.inference_mode():
        for run in selected_rows:
            checkpoint = torch.load(ROOT / run["checkpoint"], map_location="cpu", weights_only=True)
            state = checkpoint["state_dict"]
            adapted_images = F.normalize(images + images @ state["image.weight"].T, dim=-1)
            adapted_texts = F.normalize(texts + texts @ state["text.weight"].T, dim=-1)
            scores = adapted_images.double() @ adapted_texts.double().T
            winners = scores.argmax(dim=1)
            positive = (relations == 1) | (relations == 2)
            r1 = positive[torch.arange(len(image_indices)), winners].double().numpy()
            expected = run["expected_validation"]["per_image"]
            for index, iid in enumerate(validation_ids):
                supported = scores[index, relations[index] == 2]
                contradicted = scores[index, relations[index] == 3]
                accuracy = None
                if supported.numel() and contradicted.numel():
                    difference = supported[:, None] - contradicted[None, :]
                    accuracy = float(((difference > 0).double() + .5 * (difference == 0).double()).mean())
                error = abs(float(r1[index]) - expected[index]["known_positive_i2t_r1"])
                if accuracy is not None and expected[index]["relation_accuracy"] is not None:
                    error = max(error, abs(accuracy - expected[index]["relation_accuracy"]))
                elif accuracy != expected[index]["relation_accuracy"]:
                    error = 1.
                max_state_error = max(max_state_error, error)
                winner_id = manifest["texts"][text_indices[int(winners[index])]]["id"]
                if error > 1e-14 or winner_id != expected[index]["winning_text_id"]:
                    state_errors.append({"method": run["method"], "seed": run["seed"], "image_id": iid, "metric_error": error, "winner_matches": winner_id == expected[index]["winning_text_id"]})
    check("independent_selected_state_validation", not state_errors, checkpoints=36, validation_images_per_checkpoint=len(validation_ids), max_per_image_metric_error=max_state_error, errors=state_errors)
    summary = {
        "schema_version": 1, "status": "passed" if not failures else "failed",
        "audit_scope": "New reconstructed training artifacts only; no historical scores or held-out outcomes read",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "auditor_source_sha256": digest(Path(__file__)), "ledger_sha256": ledger["ledger_sha256"],
        "selection_sha256": digest(args.study / "selection.json"),
        "checks": checks, "failure_count": len(failures),
        "selected_methods": {method: {"learning_rate": entry["learning_rate"], "epochs": {str(run["seed"]): run["epoch"] for run in entry["runs"]}} for method, entry in selection["methods"].items()},
        "selected_epoch_zero_count": sum(run["epoch"] == 0 for run in selected_rows),
        "held_out_exclusion_evidence": {
            "image_splits_disjoint": not any(split_overlap.values()),
            "training_id_ledger": "train split only",
            "all_history_images": "validation split only",
            "all_validation_winners": "validation candidate pool only",
            "independent_recomputation": "36 selected states, validation features and labels only",
            "source_review": "frozen fit uses split_indices['train']; frozen validate_adapter uses split_indices['validation']; no benchmark evaluation imports",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failures": len(failures), "candidate_count": len(observed_grid), "epochs_checked": history_epochs, "selected_count": len(selected_rows), "max_selected_metric_error": max_state_error, "output": str(args.output)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
