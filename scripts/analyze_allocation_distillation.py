#!/usr/bin/env python3
"""Audit raw AD predictions and report every predeclared exploratory contrast."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gcr.allocation_distillation_analysis import (ENCODERS, FAMILIES, DATASETS, ENDPOINTS, SEEDS,
    PRIMARY_FAMILY_SIZE, DECOMPOSITION_FAMILY_SIZE, REPLICATES, BOOTSTRAP_SEED, EVIDENCE_TYPE,
    validate_evaluation_specification, equal_seed_draw_values, effect, decomposition_arrays, practical_gate)
from analyze_study import independent_metrics, metric_at
from evaluate_study import sha256, write_json
from evaluate_allocation_distillation import path_at_root, planned_states, validate_manifest

SOURCE_PATHS = ("scripts/analyze_allocation_distillation.py", "scripts/analyze_study.py",
    "scripts/evaluate_allocation_distillation.py", "scripts/evaluate_study.py",
    "src/gcr/allocation_distillation_analysis.py", "src/gcr/evaluation.py")


def load_index(repository, path, protocol_hash):
    index = json.loads(path.read_text())
    if index.get("status") != "complete" or index.get("protocol_sha256") != protocol_hash:
        raise ValueError("Analysis requires a complete protocol-bound evaluation index")
    if index.get("evidence_type") != EVIDENCE_TYPE or index.get("encoder") not in ENCODERS or set(index["datasets"]) != set(DATASETS):
        raise ValueError("Unexpected evidence type, encoder or dataset set")
    for name, digest in index["source_hashes"].items():
        if sha256(repository / name) != digest:
            raise ValueError(f"Scoring implementation changed: {name}")
    manifest_path = path_at_root(repository, index["manifest"])
    if sha256(manifest_path) != index["manifest_sha256"]:
        raise ValueError("Scored manifest content changed")
    manifest = validate_manifest(repository, manifest_path, protocol_hash)
    receipt_path = path_at_root(repository, index["prescore_receipt"])
    if sha256(receipt_path) != index["prescore_receipt_sha256"]:
        raise ValueError("Pre-score receipt content changed")
    receipt = json.loads(receipt_path.read_text())
    planned = planned_states(manifest)
    actual = [{key: value for key,value in row.items() if key != "datasets"} for row in index["runs"]]
    if actual != planned or receipt["runs"] != planned:
        raise ValueError("Scored states differ from the locked pre-score plan")
    for key in ("protocol_sha256", "encoder", "manifest_sha256", "selection_lock_sha256", "source_hashes", "selections", "decomposition_selections", "evidence_type"):
        if index[key] != receipt[key]:
            raise ValueError(f"Index differs from immutable receipt: {key}")
    predictions, metrics, raw_rows, discrepancies = {}, {}, [], []
    for run in index["runs"]:
        if set(run["datasets"]) != set(DATASETS):
            raise ValueError("Every planned state must have every benchmark")
        metrics[run["state_id"]] = {}
        for dataset, record in run["datasets"].items():
            archive = path_at_root(repository, record["predictions"])
            metadata_path = path_at_root(repository, record["metadata"])
            if sha256(archive) != record["predictions_sha256"] or sha256(metadata_path) != record["metadata_sha256"]:
                raise ValueError("Prediction artifact content hash mismatch")
            metadata = json.loads(metadata_path.read_text())
            state = {key:value for key,value in run.items() if key != "datasets"}
            expected_provenance = {"run": state, "dataset": dataset,
                                   "prescore_receipt_sha256": index["prescore_receipt_sha256"]}
            if metadata["provenance"] != expected_provenance or metadata["metrics"] != record["metrics"] or metadata["predictions_sha256"] != record["predictions_sha256"]:
                raise ValueError("Prediction metadata provenance mismatch")
            with np.load(archive, allow_pickle=False) as raw:
                prediction = {key: raw[key] for key in raw.files}
            predictions[(run["state_id"], dataset)] = prediction
            recomputed = independent_metrics(dataset, prediction)
            if dataset in ("e_vil_test1000", "coco_karpathy"):
                recomputed["mean_bidirectional_r1"] = .5 * (recomputed["i2t.r1"] + recomputed["t2i.r1"])
            for key,value in recomputed.items():
                discrepancy = abs(value - metric_at(record["metrics"], key))
                discrepancies.append(discrepancy)
                if discrepancy > 1e-12:
                    raise ValueError(f"Raw aggregate differs from scored metric: {run['state_id']} {dataset} {key}")
                metrics[run["state_id"]][f"{dataset}.{key}"] = value
                raw_rows.append({"encoder": index["encoder"], "state_id": run["state_id"],
                    "method": run["method"], "seed": run["seed"], "draw_id": run["draw_id"],
                    "learning_rate": run["learning_rate"], "epoch": run["epoch"], "source_mix": run["source_mix"],
                    "beta": run["beta"], "alpha": run["alpha"], "dataset": dataset, "metric": key, "value": value})
    return index, predictions, metrics, raw_rows, discrepancies


def selected_runs(index, family, tolerance=1.):
    states = {row["state_id"]: row for row in index["runs"]}
    if family == "frozen":
        return [states["frozen"]]
    rows = [row for row in index["selections"] if row["family"] == family and row["tolerance_pp"] == tolerance]
    expected = 9 if family == "matched_allocation_distillation" else 3
    if len(rows) != expected:
        raise ValueError("Incomplete requested selected strategy")
    result = [states[row["state_id"]] for row in rows]
    for selection, state in zip(rows, result):
        if selection["seed"] != state["seed"] or selection.get("draw_id") != state.get("draw_id"):
            raise ValueError("Selection metadata differs from its state")
    return sorted(result, key=lambda row: (row["seed"], -1 if row["draw_id"] is None else row["draw_id"]))


def strategy(index, predictions, family, dataset, metric, tolerance=1.):
    return equal_seed_draw_values(selected_runs(index, family, tolerance), predictions, dataset, metric,
        frozen=family == "frozen", expected_draws=(0,1,2) if family == "matched_allocation_distillation" else None)


def decomposition_runs(index):
    states = {row["state_id"]: row for row in index["runs"]}
    joint = {row["seed"]:row for row in selected_runs(index, "allocation_distillation")}
    cells = {name: [] for name in ("supported", "allocation", "distilled", "allocation_distillation")}
    for row in index["decomposition_selections"]:
        cell = row["cell"]
        if cell not in cells:
            raise ValueError("Unknown decomposition cell")
        state = states[row["state_id"]]
        reference = joint[row["seed"]]
        if any(state[key] != reference[key] for key in ("seed", "learning_rate", "epoch")):
            raise ValueError("Decomposition cells must share the AD-selected schedule")
        expected_mix = reference["source_mix"] if cell in ("allocation", "allocation_distillation") else 0.
        expected_beta = reference["beta"] if cell in ("distilled", "allocation_distillation") else 0.
        if state["source_mix"] != expected_mix or state["beta"] != expected_beta or state["family"] != cell or state["draw_id"] is not None:
            raise ValueError("Decomposition cells do not match the AD-selected parameters")
        cells[cell].append(state)
    if any(len(rows) != 3 or {r["seed"] for r in rows} != set(SEEDS) for rows in cells.values()):
        raise ValueError("Every decomposition cell requires all three seeds")
    return cells


def validate_matched_controls(index):
    joint = {row["seed"]: row for row in selected_runs(index, "allocation_distillation")}
    controls = selected_runs(index, "matched_allocation_distillation")
    for row in controls:
        if any(row[key] != joint[row["seed"]][key] for key in ("learning_rate", "epoch", "source_mix", "beta")):
            raise ValueError("Matched random controls must use the exact primary AD schedule and parameters")
    if {(row["seed"], row["draw_id"]) for row in controls} != {(seed, draw) for seed in SEEDS for draw in range(3)}:
        raise ValueError("All nine matched random draws must receive equal weight")


def describe_strategy(runs, metrics):
    keys = set.intersection(*(set(metrics[run["state_id"]]) for run in runs))
    summaries = {}
    for key in sorted(keys):
        if runs[0]["method"] == "frozen":
            seed_means = [metrics[runs[0]["state_id"]][key]]
        else:
            seed_means = [float(np.mean([metrics[r["state_id"]][key] for r in runs if r["seed"] == seed])) for seed in SEEDS]
        summaries[key] = {"mean": float(np.mean(seed_means)), "seed_means": seed_means,
                          "seed_std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None}
    return {"states": [{k:r[k] for k in ("state_id", "seed", "draw_id", "learning_rate", "epoch", "source_mix", "beta", "alpha", "update_norm")} for r in runs],
            "metrics": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs=2, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = path_at_root(ROOT, args.protocol)
    args.output = path_at_root(ROOT, args.output)
    args.indices = [path_at_root(ROOT, path) for path in args.indices]
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Frozen protocol content hash mismatch")
    validate_evaluation_specification(json.loads(args.protocol.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "protocol_sha256": args.protocol_sha256,
        "indices": {str(path.relative_to(ROOT)): sha256(path) for path in args.indices},
        "source_hashes": {name:sha256(ROOT / name) for name in SOURCE_PATHS},
        "evidence_type": EVIDENCE_TYPE, "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED, "primary_family_size": PRIMARY_FAMILY_SIZE,
        "decomposition_family_size": DECOMPOSITION_FAMILY_SIZE,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    receipt_path = args.output / "preanalysis_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Existing analysis receipt is immutable; use another output directory")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    data, all_raw, discrepancies = {}, [], []
    for path in args.indices:
        index, predictions, metrics, raw, diffs = load_index(ROOT, path, args.protocol_sha256)
        encoder = index["encoder"]
        if encoder in data:
            raise ValueError("Duplicate encoder index")
        data[encoder] = (index, predictions, metrics)
        all_raw.extend(raw)
        discrepancies.extend(diffs)
        validate_matched_controls(index)
    if set(data) != set(ENCODERS) or len({entry[0]["selection_lock_sha256"] for entry in data.values()}) != 1:
        raise ValueError("Both encoders must share one locked selection plan")
    with (args.output / "all_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_raw[0]))
        writer.writeheader()
        writer.writerows(all_raw)
    primary, decomposition, samples, descriptions, gates = [], [], {}, [], []
    for encoder in ENCODERS:
        index, predictions, metrics = data[encoder]
        for tolerance in (1., 0.):
            families = FAMILIES + (("matched_allocation_distillation",) if tolerance == 1. else ())
            for family in families:
                descriptions.append({"encoder": encoder, "family": family, "tolerance_pp": tolerance,
                    "inference": "descriptive", **describe_strategy(selected_runs(index, family, tolerance), metrics)})
        descriptions.append({"encoder": encoder, "family": "frozen", "tolerance_pp": None,
                             "inference": "descriptive", **describe_strategy(selected_runs(index, "frozen"), metrics)})
        comparisons = [(family, "frozen") for family in FAMILIES]
        comparisons += [("allocation_distillation", family) for family in ("distilled", "allocation", "wise_ft", "matched_allocation_distillation")]
        for left, right in comparisons:
            for dataset, metric in ENDPOINTS:
                result, draws = effect(strategy(index,predictions,left,dataset,metric),
                    strategy(index,predictions,right,dataset,metric), family_size=PRIMARY_FAMILY_SIZE)
                key = f"{encoder}__{left}_minus_{right}__{dataset}__{metric}"
                result.update({"effect_id": key, "encoder": encoder, "left": left, "right": right,
                               "dataset": dataset, "metric": metric, "tolerance_pp": 1.})
                primary.append(result)
                samples[key] = draws
                print(json.dumps({"completed_effect": key, "difference": result["difference"]}), flush=True)
        cells = decomposition_runs(index)
        for dataset,metric in ENDPOINTS:
            arrays = {name:equal_seed_draw_values(rows,predictions,dataset,metric) for name,rows in cells.items()}
            for name, values in decomposition_arrays(arrays["supported"], arrays["allocation"], arrays["distilled"], arrays["allocation_distillation"]).items():
                zero = (np.zeros_like(values[0]), values[1], values[2])
                result, draws = effect(values, zero, family_size=DECOMPOSITION_FAMILY_SIZE)
                key = f"{encoder}__{name}__{dataset}__{metric}"
                result.update({"effect_id": key, "encoder": encoder, "effect": name, "dataset": dataset,
                    "metric": metric, "conditioning": "AD development-selected learning rate, parameters, and per-seed epochs; fixed training seeds averaged before image-cluster resampling",
                    "cells": {name:[r["state_id"] for r in runs] for name,runs in cells.items()}})
                decomposition.append(result)
                samples[key] = draws
        for family in FAMILIES:
            rows = [r for r in primary if r["encoder"] == encoder and r["left"] == family and r["right"] == "frozen"]
            gates.append({"encoder": encoder, "family": family, **practical_gate(selected_runs(index,family),rows)})
    if len(primary) != PRIMARY_FAMILY_SIZE or len(decomposition) != DECOMPOSITION_FAMILY_SIZE:
        raise ValueError("Inference family sizes differ from the frozen design")
    common = {"protocol_sha256": args.protocol_sha256, "evidence_type": EVIDENCE_TYPE,
              "scale": "fractions; multiply differences and intervals by100 for percentage points",
              "test_based_selection": False, "fresh_confirmatory_claim": False}
    write_json(args.output / "primary_contrasts.json", {**common, "family_size": PRIMARY_FAMILY_SIZE, "contrasts": primary})
    write_json(args.output / "decomposition_contrasts.json", {**common, "family_size": DECOMPOSITION_FAMILY_SIZE, "contrasts": decomposition})
    write_json(args.output / "selected_strategy_metrics.json", {**common, "strategies": descriptions})
    cross_encoder = [{"family": family, "passed_both_encoders": all(row["passed"] for row in gates if row["family"] == family)} for family in FAMILIES]
    write_json(args.output / "practical_success_gates.json", {**common, "gates": gates, "cross_encoder": cross_encoder})
    np.savez_compressed(args.output / "bootstrap_samples.npz", **samples)
    audit = {"status": "passed", **common, "preanalysis_receipt_sha256": sha256(receipt_path),
        "encoders": list(ENCODERS), "unique_states": sum(len(x[0]["runs"]) for x in data.values()),
        "prediction_archives": sum(len(x[1]) for x in data.values()), "aggregate_checks": len(discrepancies),
        "maximum_recomputation_difference": max(discrepancies), "primary_effects": len(primary),
        "decomposition_effects": len(decomposition), "bootstrap_replicates": REPLICATES,
        "historical_aggregates_used": False, "all_random_draws_equally_weighted": True,
        "scope": "raw-score correctness and aggregate audit; does not independently recompute feature dot products or retrieval ranks"}
    write_json(args.output / "analysis_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
