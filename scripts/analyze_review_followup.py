#!/usr/bin/env python3
"""Audit follow-up raw predictions and report every fixed-LR/seed/draw cell.

The primary family is 18 epoch-10 effects: three prespecified contrasts at each
of three learning rates on full e-ViL test I2T/T2I R@1. Random assignment draws
are averaged within training seed. Image-cluster intervals condition on these
runs and never treat assignment draws as independent training replications.
"""
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
from evaluate_study import sha256, write_json
from analyze_study import independent_metrics, metric_at
from evaluate_review_followup import path_at_root, validate_protocol

SEEDS = (17, 29, 43)
LRS = (0.0001, 0.0003, 0.001)
DRAWS = (0, 1, 2)
METHODS = ("source", "supported") + tuple(f"{kind}_draw_{draw}" for kind in ("count_only", "score_stratified") for draw in DRAWS)
PRIMARY_REPLICATES = 10000
PRIMARY_BOOTSTRAP_SEED = 20261004
PRIMARY_FAMILY = 18
PRIMARY_DATASET = "e_vil_test1000"


def merge_indices(paths, protocol_hash):
    states, selections, input_hashes = {}, {}, {}
    for path in paths:
        index = json.loads(path.read_text())
        if index.get("status") != "complete" or index.get("evidence_type") != "review_followup_new_execution":
            raise ValueError(f"Incomplete or wrong evidence source: {path}")
        if index["protocol_sha256"] != protocol_hash:
            raise ValueError("Evaluation indices have different protocols")
        receipt_path = ROOT / index["prescore_receipt"]
        if sha256(receipt_path) != index["prescore_receipt_sha256"]:
            raise ValueError("Pre-score receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text())
        for name, hashes in receipt["input_hashes"].items():
            if name in input_hashes and input_hashes[name] != hashes:
                raise ValueError("Dataset changed between evaluation suites")
            input_hashes[name] = hashes
        for row in index["selections"]:
            key = (row["selector"], row["condition"], row["draw_id"], row["seed"])
            if key in selections and selections[key] != row:
                raise ValueError("Selection changed between evaluation suites")
            selections[key] = row
        for row in index["runs"]:
            sid = row["state_id"]
            if sid not in states:
                states[sid] = row
            else:
                old = states[sid]
                if {k: v for k, v in old.items() if k != "datasets"} != {k: v for k, v in row.items() if k != "datasets"}:
                    raise ValueError("Checkpoint metadata changed between suites")
                for name, value in row["datasets"].items():
                    if name in old["datasets"] and old["datasets"][name] != value:
                        raise ValueError("Prediction changed between evaluation suites")
                    old["datasets"][name] = value
    return states, list(selections.values()), input_hashes


def aligned_values(runs, predictions, dataset, metric):
    """Return rows in supplied run order and exact item/cluster identities."""
    values, first_ids, first_clusters = [], None, None
    for run in runs:
        p = predictions[(run["state_id"], dataset)]
        if metric in ("i2t.r1", "t2i.r1"):
            direction = metric[:3]
            ids = p["image_ids"] if direction == "i2t" else p["text_ids"]
            clusters = p["image_ids"] if direction == "i2t" else p["text_source_image_ids"]
            value = (p[f"{direction}_ranks"] <= 1).astype(np.float64)
        elif dataset == "visual_entailment" and metric == "accuracy":
            ids, clusters, value = p["image_ids"], p["image_ids"], p["image_accuracy"]
        else:
            key = {"accuracy": "correct", "positive1_accuracy": "positive1_correct", "positive2_accuracy": "positive2_correct", "both_accuracy": "correct"}[metric]
            ids, clusters, value = p["item_ids"], p["image_ids"], p[key].astype(np.float64)
        if first_ids is None:
            first_ids, first_clusters = ids, clusters
        elif not np.array_equal(ids, first_ids) or not np.array_equal(clusters, first_clusters):
            raise ValueError("Paired prediction identities differ")
        values.append(value)
    return np.stack(values), first_ids, first_clusters


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "values": values.tolist(),
            "std": float(values.std(ddof=1)) if values.size > 1 else None,
            "min": float(values.min()), "max": float(values.max())}


def policy_group(method):
    if method.startswith("count_only_draw_"):
        return "count_only"
    if method.startswith("score_stratified_draw_"):
        return "score_stratified"
    return method


def expected_methods(group):
    return [f"{group}_draw_{draw}" for draw in DRAWS] if group in ("count_only", "score_stratified") else [group]


def grid_summaries(runs, metrics):
    """All method cells and equal-draw policy means, with both sources of variation."""
    cells = {}
    for row in runs:
        if row["method"] == "frozen":
            continue
        key = (row["method"], row["learning_rate"], row["epoch"])
        cells.setdefault(key, []).append(row)
    summaries = []
    for (method, lr, epoch), rows in sorted(cells.items()):
        rows = sorted(rows, key=lambda row: row["seed"])
        common = set.intersection(*(set(metrics[row["state_id"]]) for row in rows))
        entry = {"method": method, "condition": policy_group(method), "learning_rate": lr, "epoch": epoch,
                 "seeds": [row["seed"] for row in rows], "state_ids": [row["state_id"] for row in rows],
                 "complete_seed_cell": tuple(row["seed"] for row in rows) == SEEDS, "metrics": {}}
        for key in sorted(common):
            entry["metrics"][key] = summarize([metrics[row["state_id"]][key] for row in rows])
        summaries.append(entry)
    pooled = []
    for group in ("count_only", "score_stratified"):
        for lr in LRS:
            for epoch in (1, 5, 10):
                entries = [row for row in summaries if row["condition"] == group and row["learning_rate"] == lr and row["epoch"] == epoch]
                if len(entries) != 3 or not all(row["complete_seed_cell"] for row in entries):
                    continue
                entries.sort(key=lambda row: row["method"])
                keys = set.intersection(*(set(row["metrics"]) for row in entries))
                result = {"condition": group, "learning_rate": lr, "epoch": epoch, "draws": list(DRAWS),
                          "seeds": list(SEEDS), "weighting": "equal draws within seed, equal training seeds", "metrics": {}}
                for key in sorted(keys):
                    matrix = np.array([row["metrics"][key]["values"] for row in entries])
                    result["metrics"][key] = {"mean": float(matrix.mean()), "draw_by_seed_values": matrix.tolist(),
                                              "per_seed_means": summarize(matrix.mean(axis=0)),
                                              "per_draw_means": summarize(matrix.mean(axis=1))}
                pooled.append(result)
    return summaries, pooled


def primary_analysis(states, predictions):
    lookup = {(row["method"], row["learning_rate"], row["seed"]): row for row in states.values() if row["epoch"] == 10}
    expected = {(method, lr, seed) for method in METHODS for lr in LRS for seed in SEEDS}
    if set(lookup) != expected:
        raise ValueError(f"Primary terminal grid must contain exactly 72 states: missing {expected - set(lookup)}, extra {set(lookup) - expected}")
    effects, arrays = [], {}
    for lr in LRS:
        support_rows = [lookup[("supported", lr, seed)] for seed in SEEDS]
        for comparator in ("source", "count_only", "score_stratified"):
            for metric in ("i2t.r1", "t2i.r1"):
                support, ids, clusters = aligned_values(support_rows, predictions, PRIMARY_DATASET, metric)
                per_draw = []
                for method in expected_methods(comparator):
                    values, other_ids, other_clusters = aligned_values([lookup[(method, lr, seed)] for seed in SEEDS], predictions, PRIMARY_DATASET, metric)
                    if not np.array_equal(ids, other_ids) or not np.array_equal(clusters, other_clusters):
                        raise ValueError("Primary comparator query identities differ")
                    per_draw.append(values)
                comparator_values = np.stack(per_draw).mean(axis=0)
                effect = paired_image_bootstrap(support, comparator_values, clusters, replicates=PRIMARY_REPLICATES,
                                                seed=PRIMARY_BOOTSTRAP_SEED, family_size=PRIMARY_FAMILY)
                effect_id = f"supported_minus_{comparator}__lr_{lr:g}__{metric.replace('.', '_')}"
                arrays[effect_id] = effect.pop("bootstrap_differences")
                effect.update(effect_id=effect_id, learning_rate=lr, epoch=10, dataset=PRIMARY_DATASET, metric=metric,
                              left="supported", right=comparator, seed_ids=list(SEEDS),
                              comparator_draw_methods=expected_methods(comparator),
                              comparator_draw_by_seed_values=np.asarray(per_draw).mean(axis=2).tolist(),
                              supported_seed_values=support.mean(axis=1).tolist(),
                              difference_percentage_points=100 * effect["difference"],
                              ci_percentage_points=[100 * effect["ci_lower"], 100 * effect["ci_upper"]],
                              interpretation="conditional on the fitted training seeds and fixed assignment draws; neither is resampled")
                effects.append(effect)
    if len(effects) != PRIMARY_FAMILY:
        raise AssertionError("Primary family size differs from frozen design")
    return effects, arrays


def selected_draw_means(entries):
    results = []
    for selector in sorted({row["selector"] for row in entries}):
        for group in ("count_only", "score_stratified"):
            rows = [row for row in entries if row["selector"] == selector and row["condition"] == group]
            if len(rows) != 3:
                raise ValueError("Selected strategy lacks one or more assignment draws")
            rows.sort(key=lambda row: row["method"])
            common = set.intersection(*(set(row["metrics"]) for row in rows))
            result = {"selector": selector, "condition": group, "draw_ids": list(DRAWS), "seed_ids": list(SEEDS), "metrics": {}}
            for key in sorted(common):
                values = np.asarray([row["metrics"][key]["values"] for row in rows])
                result["metrics"][key] = {"mean": float(values.mean()), "draw_by_seed_values": values.tolist(),
                                          "per_seed_means": summarize(values.mean(axis=0)), "per_draw_means": summarize(values.mean(axis=1))}
            results.append(result)
    return results


def subgroup_summaries(states, predictions, dataset_hashes):
    source_path = ROOT / "data/visual_entailment/manifest.json"
    if sha256(source_path) != dataset_hashes["visual_entailment"]["manifest_sha256"]:
        raise ValueError("Original relation split differs from scored manifest")
    manifest = json.loads(source_path.read_text())
    original = {str(row["id"]) for row in manifest["images"] if row["split"] == "test"}
    if len(original) != 400:
        raise ValueError("Original test subgroup must have 400 images")
    results = []
    for sid, state in sorted(states.items()):
        key = (sid, PRIMARY_DATASET)
        if key not in predictions:
            continue
        p = predictions[key]
        if not original.issubset(set(p["image_ids"])):
            raise ValueError("Full e-ViL pool lacks original test images")
        for group in ("original400", "additional600"):
            row = {"state_id": sid, "method": state["method"], "learning_rate": state["learning_rate"],
                   "epoch": state["epoch"], "seed": state["seed"], "draw_id": state["draw_id"], "subgroup": group, "metrics": {}}
            for direction, cluster_key in (("i2t", "image_ids"), ("t2i", "text_source_image_ids")):
                mask = np.asarray([str(iid) in original for iid in p[cluster_key]])
                if group == "additional600":
                    mask = ~mask
                ranks = p[f"{direction}_ranks"][mask]
                expected = (400 if group == "original400" else 600) * (1 if direction == "i2t" else 5)
                if len(ranks) != expected:
                    raise ValueError("Unexpected subgroup query count")
                row["metrics"][direction] = {"queries": len(ranks), "r1": float(np.mean(ranks <= 1)),
                                              "r5": float(np.mean(ranks <= 5)), "r10": float(np.mean(ranks <= 10)),
                                              "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks))}
            results.append(row)
    return {"status": "descriptive query subgroups, no additional inferential family", "candidate_pool": "same full 1000 images / 5000 captions for both subgroups", "records": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/review_followup/analysis")
    args = parser.parse_args()
    args.indices = [path_at_root(path) for path in args.indices]
    args.protocol = path_at_root(args.protocol)
    args.output = path_at_root(args.output)
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Protocol differs from explicitly frozen analysis")
    validate_protocol(json.loads(args.protocol.read_text()))
    states, selections, dataset_hashes = merge_indices(args.indices, args.protocol_sha256)
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__).resolve(), ROOT / "scripts/analyze_study.py", ROOT / "scripts/evaluate_review_followup.py", ROOT / "scripts/evaluate_study.py", ROOT / "src/gcr/evaluation.py"]
    receipt = {"schema_version": 1, "protocol_sha256": args.protocol_sha256,
               "evaluation_indices": {str(path.relative_to(ROOT)): sha256(path) for path in args.indices},
               "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
               "primary_family": PRIMARY_FAMILY, "bootstrap_seed": PRIMARY_BOOTSTRAP_SEED,
               "bootstrap_replicates": PRIMARY_REPLICATES, "dataset_hashes": dataset_hashes}
    receipt_path = args.output / "preanalysis_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Analysis inputs/source differ from preanalysis receipt; use another output directory")
    write_json(receipt_path, receipt)
    predictions, metrics, differences, raw_rows = {}, {}, [], []
    for sid, state in sorted(states.items()):
        metrics[sid] = {}
        for dataset, record in state["datasets"].items():
            path = ROOT / record["predictions"]
            if sha256(path) != record["predictions_sha256"]:
                raise ValueError(f"Prediction archive hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as raw:
                p = {key: raw[key] for key in raw.files}
            predictions[(sid, dataset)] = p
            recomputed = independent_metrics(dataset, p)
            if dataset in ("e_vil_test1000", "coco_karpathy"):
                recomputed["mean_bidirectional_r1"] = 0.5 * (recomputed["i2t.r1"] + recomputed["t2i.r1"])
            for key, value in recomputed.items():
                discrepancy = abs(value - metric_at(record["metrics"], key))
                differences.append(discrepancy)
                if discrepancy > 1e-12:
                    raise ValueError(f"Raw prediction aggregate mismatch: {sid} {dataset} {key}")
                label = f"{dataset}.{key}"
                metrics[sid][label] = value
                raw_rows.append({"state_id": sid, "condition": state["condition"], "method": state["method"], "draw_id": state["draw_id"],
                                 "learning_rate": state["learning_rate"], "seed": state["seed"], "epoch": state["epoch"],
                                 "dataset": dataset, "metric": key, "value": value})
    with (args.output / "all_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    fixed_rows = [row for row in states.values() if row["epoch"] in (1, 5, 10)]
    grid, pooled = grid_summaries(fixed_rows, metrics)
    write_json(args.output / "fixed_epoch_metrics.json", {"status": "all cells reported; no held-out model selection", "cells": grid, "equal_draw_policy_means": pooled})
    effects, arrays = primary_analysis(states, predictions)
    write_json(args.output / "primary_contrasts.json", {"family": "3 contrasts × 3 learning rates × 2 source-only Flickr retrieval endpoints",
                                                        "confidence": 1 - .05 / PRIMARY_FAMILY, "contrasts": effects,
                                                        "protocol_sha256": args.protocol_sha256})
    np.savez_compressed(args.output / "primary_bootstrap_samples.npz", **arrays)
    selected_output = []
    for selector in sorted({row["selector"] for row in selections}):
        selector_rows = [row for row in selections if row["selector"] == selector]
        for method in METHODS:
            chosen = [states[row["state_id"]] for row in selector_rows if row["state_id"] in states and states[row["state_id"]]["method"] == method]
            if tuple(sorted(row["seed"] for row in chosen)) != SEEDS:
                raise ValueError(f"Missing selected states for {selector} {method}")
            chosen.sort(key=lambda row: row["seed"])
            common = set.intersection(*(set(metrics[row["state_id"]]) for row in chosen))
            selected_output.append({"selector": selector, "method": method, "condition": policy_group(method),
                                    "selected_states": [{key: row[key] for key in ("state_id", "seed", "learning_rate", "epoch", "draw_id")} for row in chosen],
                                    "metrics": {key: summarize([metrics[row["state_id"]][key] for row in chosen]) for key in sorted(common)}})
    write_json(args.output / "selected_strategy_metrics.json", {"status": "descriptive complete selection strategies; no winner-draw selection",
                                                                "strategies": selected_output,
                                                                "equal_draw_policy_means": selected_draw_means(selected_output),
                                                                "interpretation": "Native and source-retrieval strategies differ jointly in criterion, relevance, pool size, development images and directions."})
    write_json(args.output / "e_vil_query_subgroups.json", subgroup_summaries(states, predictions, dataset_hashes))
    audit = {"status": "passed", "protocol_sha256": args.protocol_sha256, "preanalysis_receipt_sha256": sha256(receipt_path),
             "unique_states": len(states), "prediction_archives": len(predictions), "aggregate_checks": len(differences),
             "maximum_recomputation_difference": max(differences), "primary_effects": len(effects),
             "historical_aggregates_used": False, "test_based_selection": False}
    write_json(args.output / "analysis_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
