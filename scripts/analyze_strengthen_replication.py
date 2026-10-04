#!/usr/bin/env python3
"""Audit complete replication predictions and the original twelve contrasts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from gcr.evaluation import paired_image_bootstrap
from gcr.training import atomic_json, sha256_file
from gcr.review_training import _atomic_npz, file_record
from analyze_study import independent_metrics, metric_at
from analyze_review_followup import aligned_values, grid_summaries, summarize
from evaluate_strengthen_replication import (DATASETS, EVIDENCE_TYPE, POLICIES,
    RATES, SEEDS, SELECTORS, PROTOCOL_SHA256, at_root, validate_protocol)


def primary_analysis(states: dict, predictions: dict, protocol: dict):
    rows = [row for row in states.values() if row["epoch"] == 10]
    lookup = {(row["method"], row["learning_rate"], row["seed"]): row for row in rows}
    expected = {(method, rate, seed) for method in POLICIES for rate in RATES for seed in SEEDS}
    if len(rows) != 45 or len(lookup) != 45 or set(lookup) != expected:
        raise ValueError("The primary grid must contain exactly 45 terminal states")
    p = protocol["primary_analysis"]
    effects, samples = [], {}
    for rate in RATES:
        left = [lookup[("supported", rate, seed)] for seed in SEEDS]
        for comparator in ("source", "score_stratified"):
            methods = ["source"] if comparator == "source" else list(POLICIES[2:])
            for metric in ("i2t.r1", "t2i.r1"):
                a, ids, clusters = aligned_values(left, predictions, "e_vil_test1000", metric)
                drawn = []
                for method in methods:
                    b, other_ids, other_clusters = aligned_values(
                        [lookup[(method, rate, seed)] for seed in SEEDS],
                        predictions, "e_vil_test1000", metric)
                    if not np.array_equal(ids, other_ids) or not np.array_equal(clusters, other_clusters):
                        raise ValueError("Primary query or cluster identities differ")
                    drawn.append(b)
                average_control = np.stack(drawn).mean(axis=0)
                effect = paired_image_bootstrap(a, average_control, clusters,
                    replicates=p["bootstrap_replicates"], seed=p["bootstrap_seed"],
                    family_size=p["family_size"], alpha=p["alpha"])
                effect_id = f"supported_minus_{comparator}__lr_{rate:g}__{metric.replace('.', '_')}"
                samples[effect_id] = effect.pop("bootstrap_differences")
                effect.update(effect_id=effect_id, learning_rate=rate, epoch=10,
                    dataset="e_vil_test1000", metric=metric, left="supported", right=comparator,
                    seed_ids=list(SEEDS), comparator_draw_methods=methods,
                    comparator_draw_by_seed_values=np.stack(drawn).mean(axis=2).tolist(),
                    supported_seed_values=a.mean(axis=1).tolist(),
                    difference_percentage_points=100 * effect["difference"],
                    ci_percentage_points=[100 * effect["ci_lower"], 100 * effect["ci_upper"]],
                    interpretation="conditional on fitted seeds, assignment draws, and fixed gallery")
                effects.append(effect)
    if len(effects) != 12:
        raise AssertionError("The frozen family must contain twelve contrasts")
    return effects, samples


def selected_summaries(states, selections, metrics):
    """Preserve seed identities when several selections map to frozen scores."""
    strategies = []
    for selector in SELECTORS:
        for method in POLICIES:
            selected = sorted([row for row in selections if row["selector"] == selector
                               and row["method"] == method], key=lambda row: row["seed"])
            if tuple(row["seed"] for row in selected) != SEEDS:
                raise ValueError("Selected strategy lacks a declared optimizer seed")
            if len({row["learning_rate"] for row in selected}) != 1:
                raise ValueError("The selected rate must be common across seeds")
            rows = [metrics[row["state_id"]] for row in selected]
            if any(set(row) != set(rows[0]) for row in rows):
                raise ValueError("Selected strategy metrics are incomplete")
            strategies.append({"selector": selector, "method": method,
                "condition": "score_stratified" if method.startswith("score_stratified") else method,
                "selected_states": selected,
                "all_epochs_zero": all(row["epoch"] == 0 for row in selected),
                "metrics": {key: summarize([row[key] for row in rows]) for key in sorted(rows[0])}})
    pooled = []
    for selector in SELECTORS:
        entries = [row for row in strategies if row["selector"] == selector
                   and row["condition"] == "score_stratified"]
        entries.sort(key=lambda row: row["method"])
        if [row["method"] for row in entries] != list(POLICIES[2:]):
            raise ValueError("Every random draw must appear in the selected strategy")
        result = {"selector": selector, "condition": "score_stratified",
            "draw_ids": [0, 1, 2], "seed_ids": list(SEEDS), "metrics": {}}
        for key in entries[0]["metrics"]:
            matrix = np.array([row["metrics"][key]["values"] for row in entries])
            result["metrics"][key] = {"mean": float(matrix.mean()),
                "draw_by_seed_values": matrix.tolist(),
                "per_seed_means": summarize(matrix.mean(axis=0)),
                "per_draw_means": summarize(matrix.mean(axis=1))}
        pooled.append(result)
    return {"status": "descriptive; both full development procedures, with every random draw",
        "strategies": strategies, "equal_draw_policy_means": pooled,
        "selection_uncertainty": "not quantified; no test-based selection"}


def verify_index(index_path: Path, setting: str, protocol_hash: str):
    index = json.loads(index_path.read_text())
    if index.get("status") != "complete" or index.get("evidence_type") != EVIDENCE_TYPE:
        raise ValueError("Analysis requires the complete new replication execution")
    if index["setting"] != setting or index["protocol_sha256"] != protocol_hash:
        raise ValueError("Evaluation setting or protocol differs")
    receipt_path = ROOT / index["prescore_receipt"]
    if sha256_file(receipt_path) != index["prescore_receipt_sha256"]:
        raise ValueError("Pre-score receipt digest mismatch")
    receipt = json.loads(receipt_path.read_text())
    if receipt["selections"] != index["selections"] or receipt["setting"] != setting:
        raise ValueError("Selected states changed after scoring began")
    for name, digest in receipt["source_hashes"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Scoring source changed since scoring began: {name}")
    for name in ("state_manifest", "ledger", "dataset_config"):
        record = receipt[name]
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Scoring input changed: {name}")
    rows = index["runs"]
    states = {row["state_id"]: row for row in rows}
    if len(states) != len(rows) or len(rows) != index["run_count"]:
        raise ValueError("Duplicate or missing evaluation states")
    expected = {row["state_id"]: row for row in receipt["states"]}
    if set(states) != set(expected) or set(index["datasets"]) != set(DATASETS):
        raise ValueError("Incomplete planned states or datasets")
    for sid, row in states.items():
        if {key: value for key, value in row.items() if key != "datasets"} != expected[sid]:
            raise ValueError("Evaluated state changed from its frozen plan")
        if set(row["datasets"]) != set(DATASETS):
            raise ValueError("Each state requires every declared dataset")
    return index, receipt, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--setting", choices=PROTOCOL_SHA256, required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in ("index", "protocol", "output"):
        setattr(args, name, at_root(getattr(args, name)))
    protocol = validate_protocol(args.protocol, args.setting, args.protocol_sha256)
    index, receipt, states = verify_index(args.index, args.setting, args.protocol_sha256)
    names = ("scripts/analyze_strengthen_replication.py", "scripts/evaluate_strengthen_replication.py",
        "scripts/analyze_study.py", "scripts/analyze_review_followup.py", "src/gcr/evaluation.py")
    analysis_receipt = {"schema_version": 1, "setting": args.setting,
        "protocol_sha256": args.protocol_sha256, "evaluation_index": file_record(ROOT, args.index),
        "source_hashes": {name: sha256_file(ROOT / name) for name in names},
        "primary_family": 12, "bootstrap_replicates": 10000,
        "bootstrap_seed": protocol["primary_analysis"]["bootstrap_seed"],
        "evidence_type": EVIDENCE_TYPE, "historical_outcomes_used": False}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "preanalysis_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != analysis_receipt:
        raise ValueError("Analysis provenance changed; use a new output directory")
    atomic_json(receipt_path, analysis_receipt)
    predictions, metrics, discrepancies, csv_rows = {}, {}, [], []
    for sid, state in sorted(states.items()):
        metrics[sid] = {}
        for name in DATASETS:
            record = state["datasets"][name]
            raw_path, metadata_path = ROOT / record["predictions"], ROOT / record["metadata"]
            if sha256_file(raw_path) != record["predictions_sha256"] or sha256_file(metadata_path) != record["metadata_sha256"]:
                raise ValueError("A prediction archive or its metadata changed")
            metadata = json.loads(metadata_path.read_text())
            if metadata["predictions_sha256"] != record["predictions_sha256"] or metadata["metrics"] != record["metrics"]:
                raise ValueError("Prediction metadata disagrees with evaluation index")
            expected_provenance = {"prescore_receipt_sha256": index["prescore_receipt_sha256"],
                "state": {key: value for key, value in state.items() if key != "datasets"},
                "dataset": name, "input_hashes": receipt["input_hashes"][name]}
            if metadata["provenance"] != expected_provenance:
                raise ValueError("Prediction belongs to another state, dataset, or scoring plan")
            with np.load(raw_path, allow_pickle=False) as archive:
                p = {key: archive[key] for key in archive.files}
            predictions[(sid, name)] = p
            if name == "e_vil_test1000":
                if len(p["image_ids"]) != 1000 or len(p["text_ids"]) != 5000:
                    raise ValueError("Primary gallery size differs from the frozen protocol")
                unique, counts = np.unique(p["text_source_image_ids"], return_counts=True)
                if set(unique) != set(p["image_ids"]) or not np.all(counts == 5):
                    raise ValueError("Every primary image must own exactly five source captions")
            values = independent_metrics(name, p)
            if name in ("e_vil_test1000", "coco_karpathy"):
                values["mean_bidirectional_r1"] = .5 * (values["i2t.r1"] + values["t2i.r1"])
            for key, value in values.items():
                error = abs(value - metric_at(record["metrics"], key))
                if not np.isfinite(error) or error > 1e-12:
                    raise ValueError(f"Raw aggregate differs: {sid}, {name}, {key}")
                discrepancies.append(error)
                metrics[sid][f"{name}.{key}"] = value
                csv_rows.append({"state_id": sid, "method": state["method"],
                    "draw_id": state["draw_id"], "learning_rate": state["learning_rate"],
                    "seed": state["seed"], "epoch": state["epoch"],
                    "dataset": name, "metric": key, "value": value})
    with (args.output / "all_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    grid, pooled = grid_summaries([row for row in states.values() if row["epoch"] == 10], metrics)
    atomic_json(args.output / "terminal_metrics.json", {"epoch": 10, "cells": grid, "equal_draw_policy_means": pooled})
    atomic_json(args.output / "frozen_metrics.json", metrics["frozen"])
    atomic_json(args.output / "selected_strategy_metrics.json", selected_summaries(states, index["selections"], metrics))
    effects, arrays = primary_analysis(states, predictions, protocol)
    atomic_json(args.output / "primary_contrasts.json", {"setting": args.setting,
        "protocol_sha256": args.protocol_sha256,
        "family": "2 contrasts x 3 rates x 2 retrieval directions", "family_size": 12,
        "confidence": 1 - .05 / 12, "contrasts": effects})
    _atomic_npz(args.output / "primary_bootstrap_samples.npz", **arrays)
    audit = {"status": "passed", "setting": args.setting, "states": len(states),
        "prediction_archives": len(predictions), "aggregate_checks": len(discrepancies),
        "maximum_recomputation_difference": max(discrepancies), "primary_effects": len(effects),
        "bootstrap_samples": sum(len(v) for v in arrays.values()),
        "preanalysis_receipt_sha256": sha256_file(receipt_path),
        "test_based_selection": False, "missing_prior_outcomes_used": False}
    atomic_json(args.output / "analysis_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
