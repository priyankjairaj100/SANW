#!/usr/bin/env python3
"""Audit new raw predictions, aggregate selected seeds and run planned intervals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from gcr.evaluation import paired_image_bootstrap

DATASETS = ("visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
METHODS = ("clip", "sanw_fixed", "sanw_median", "constant", "shuffled", "multipositive", "grounded",
           "grounded_no_hardening", "grounded_no_abstention", "pairwise_rank", "random_exclusion", "smoothing")
SEEDS = (17, 29, 43)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def independent_metrics(name: str, p: dict[str, np.ndarray]) -> dict[str, float]:
    """Recompute aggregates with simple loops independent of evaluator helpers."""
    if name == "visual_entailment":
        raw: dict[str, dict[str, list[float]]] = {}
        for iid, relation, score in zip(p["pair_image_ids"], p["pair_relations"], p["pair_scores"]):
            raw.setdefault(str(iid), {"supported": [], "contradicted": []})[str(relation)].append(float(score))
        values = []
        for iid, recorded in zip(p["image_ids"], p["image_accuracy"]):
            groups = raw[str(iid)]
            comparisons = [1.0 if a > b else 0.5 if a == b else 0.0
                           for a in groups["supported"] for b in groups["contradicted"]]
            value = sum(comparisons) / len(comparisons)
            if abs(value - float(recorded)) > 1e-12:
                raise ValueError(f"Raw hypothesis scores disagree with image accuracy: {iid}")
            values.append(value)
        return {"accuracy": sum(values) / len(values)}
    if name in ("sugarcrepe", "sugarcrepe_pp"):
        p1 = [bool(a > b) for a, b in zip(p["positive1_scores"], p["negative_scores"])]
        if not np.array_equal(p1, p["positive1_correct"]):
            raise ValueError("Raw scores disagree with first-positive correctness")
        values = {"positive1_accuracy": sum(p1) / len(p1)}
        correct = p1
        if name == "sugarcrepe_pp":
            p2 = [bool(a > b) for a, b in zip(p["positive2_scores"], p["negative_scores"])]
            if not np.array_equal(p2, p["positive2_correct"]):
                raise ValueError("Raw scores disagree with alternative-positive correctness")
            correct = [a and b for a, b in zip(p1, p2)]
            values.update({"positive2_accuracy": sum(p2) / len(p2), "both_accuracy": sum(correct) / len(correct)})
        if not np.array_equal(correct, p["correct"]):
            raise ValueError("Raw scores disagree with benchmark correctness")
        values["accuracy"] = sum(correct) / len(correct)
        return values
    values = {}
    for direction in ("i2t", "t2i"):
        ranks = p[f"{direction}_ranks"].tolist()
        for cutoff in (1, 5, 10):
            values[f"{direction}.r{cutoff}"] = sum(rank <= cutoff for rank in ranks) / len(ranks)
        values[f"{direction}.mean_rank"] = sum(ranks) / len(ranks)
        ordered = sorted(ranks)
        middle = len(ordered) // 2
        values[f"{direction}.median_rank"] = float(ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2)
    return values


def metric_at(metrics: dict, key: str) -> float:
    value = metrics
    for segment in key.split("."):
        value = value[segment]
    return float(value)


def stack_aligned(runs: list[dict], predictions: dict, dataset: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    reference = predictions[(runs[0]["run_id"], dataset)]
    ids_key = "image_ids" if dataset == "visual_entailment" else "item_ids"
    values = []
    for run in runs:
        p = predictions[(run["run_id"], dataset)]
        if not np.array_equal(p[ids_key], reference[ids_key]) or not np.array_equal(p["image_ids"], reference["image_ids"]):
            raise ValueError(f"Item order changed between selected runs: {dataset}")
        values.append(p[key])
    return np.asarray(values, dtype=np.float64), reference["image_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "results/evaluation/index.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    if not index.get("complete_selected_study") or index["status"] != "complete" or index["evidence_type"] != "new_execution":
        raise ValueError("Analysis requires all 36 selected adapters plus frozen on all four full benchmarks")
    by_method = {method: sorted([run for run in index["runs"] if run["method"] == method], key=lambda run: run["seed"] or 0)
                 for method in ("frozen",) + METHODS}
    if len(by_method["frozen"]) != 1:
        raise ValueError("Exactly one frozen reference is required")
    for method in METHODS:
        if tuple(run["seed"] for run in by_method[method]) != SEEDS:
            raise ValueError(f"Wrong selected seed set for {method}")
    args.output.mkdir(parents=True, exist_ok=True)
    predictions, recomputed, discrepancies = {}, {}, []
    for run in index["runs"]:
        recomputed[run["run_id"]] = {}
        for name in DATASETS:
            record = run["datasets"][name]
            path = ROOT / record["predictions"]
            if sha256(path) != record["predictions_sha256"]:
                raise ValueError(f"Prediction archive hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as values:
                p = {key: values[key] for key in values.files}
            predictions[(run["run_id"], name)] = p
            recalculated = independent_metrics(name, p)
            for key, value in recalculated.items():
                difference = abs(value - metric_at(record["metrics"], key))
                discrepancies.append(difference)
                if difference > 1e-12:
                    raise ValueError(f"Stored aggregate disagrees with raw predictions: {run['run_id']} {name} {key}")
            recomputed[run["run_id"]][name] = recalculated
    summary, table = {}, []
    for method, runs in by_method.items():
        summary[method] = {"runs": [{key: run.get(key) for key in ("run_id", "seed", "epoch", "learning_rate", "checkpoint_sha256")} for run in runs], "metrics": {}}
        row = {"method": method, "selected_seeds": len(runs)}
        for name in DATASETS:
            for key in recomputed[runs[0]["run_id"]][name]:
                values = [recomputed[run["run_id"]][name][key] for run in runs]
                label = f"{name}.{key}"
                mean, std = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None
                summary[method]["metrics"][label] = {"mean": mean, "seed_std": std, "seed_values": values}
                row[label + ".mean"] = mean
                row[label + ".seed_std"] = std
        table.append(row)
    write_json(args.output / "aggregate_metrics.json", {"source_index_sha256": sha256(args.index), "units": "accuracy and recall as fractions; rank one-based", "methods": summary})
    with (args.output / "aggregate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    contrasts, bootstrap_arrays = [], {}
    for comparator in ("multipositive", "grounded_no_hardening", "grounded_no_abstention"):
        for dataset, value_key in (("visual_entailment", "image_accuracy"), ("sugarcrepe", "correct")):
            a, image_ids = stack_aligned(by_method["grounded"], predictions, dataset, value_key)
            b, other_ids = stack_aligned(by_method[comparator], predictions, dataset, value_key)
            if not np.array_equal(image_ids, other_ids):
                raise ValueError("Image clusters disagree between primary methods")
            # stack_aligned verifies within-method item order; check between methods too.
            pa = predictions[(by_method["grounded"][0]["run_id"], dataset)]
            pb = predictions[(by_method[comparator][0]["run_id"], dataset)]
            id_key = "image_ids" if dataset == "visual_entailment" else "item_ids"
            if not np.array_equal(pa[id_key], pb[id_key]):
                raise ValueError("Paired primary item IDs disagree")
            result = paired_image_bootstrap(a, b, image_ids, replicates=args.replicates, seed=args.bootstrap_seed, family_size=6)
            effect_id = f"grounded_minus_{comparator}__{dataset}"
            bootstrap_arrays[effect_id] = result.pop("bootstrap_differences")
            result.update({"effect_id": effect_id, "method_a": "grounded", "method_b": comparator, "dataset": dataset,
                           "seeds": list(SEEDS), "difference_percentage_points": 100 * result["difference"],
                           "ci_percentage_points": [100 * result["ci_lower"], 100 * result["ci_upper"]]})
            contrasts.append(result)
    write_json(args.output / "primary_contrasts.json", {"source_index_sha256": sha256(args.index), "contrasts": contrasts,
                                                        "family": "six fixed contrasts across two primary endpoints", "selection_uncertainty": "not included"})
    np.savez_compressed(args.output / "primary_bootstrap_samples.npz", **bootstrap_arrays)
    category_reference = predictions[(by_method["frozen"][0]["run_id"], "sugarcrepe")]
    categories = category_reference["categories"]
    masks = {category: categories == category for category in sorted(set(categories))}
    masks.update({"added_content": np.isin(categories, ["add_obj", "add_att"]),
                  "combined_replace_swap": np.asarray([str(category).startswith(("replace_", "swap_")) for category in categories]),
                  "all": np.ones(len(categories), dtype=bool)})
    if not np.array_equal(masks["added_content"] | masks["combined_replace_swap"], masks["all"]) or np.any(masks["added_content"] & masks["combined_replace_swap"]):
        raise ValueError("Descriptive category groups must partition the full SugarCrepe benchmark")
    category_output = {"status": "post_hoc_descriptive; no extra significance tests or model selection", "groups": {}}
    for group, mask in masks.items():
        entry = {"items": int(mask.sum()), "images": len(np.unique(category_reference["image_ids"][mask])), "methods": {}}
        for method, runs in by_method.items():
            values, _ = stack_aligned(runs, predictions, "sugarcrepe", "correct")
            seed_values = values[:, mask].mean(axis=1)
            entry["methods"][method] = {"mean": float(seed_values.mean()), "seed_values": seed_values.tolist()}
        mp = np.asarray(entry["methods"]["multipositive"]["seed_values"])
        sp = np.asarray(entry["methods"]["clip"]["seed_values"])
        entry["multipositive_minus_source"] = {"mean": float(np.mean(mp - sp)), "seed_differences": (mp - sp).tolist()}
        category_output["groups"][group] = entry
    write_json(args.output / "sugarcrepe_category_descriptive.json", category_output)
    retrieval_transitions = []
    for sp_run, mp_run in zip(by_method["clip"], by_method["multipositive"]):
        sp = predictions[(sp_run["run_id"], "coco_karpathy")]
        mp = predictions[(mp_run["run_id"], "coco_karpathy")]
        entry = {"seed": sp_run["seed"]}
        for direction, ids_key in (("i2t", "image_ids"), ("t2i", "text_ids")):
            if not np.array_equal(sp[ids_key], mp[ids_key]):
                raise ValueError("Retrieval query order differs between policies")
            before, after = sp[f"{direction}_ranks"] == 1, mp[f"{direction}_ranks"] == 1
            entry[direction] = {"queries": len(before), "lost_successes": int(np.sum(before & ~after)),
                                "new_successes": int(np.sum(~before & after)),
                                "net_additional_failures": int(before.sum()) - int(after.sum()),
                                "recall_difference": float(after.mean() - before.mean())}
        retrieval_transitions.append(entry)
    retention = []
    frozen = summary["frozen"]["metrics"]
    for run in index["runs"]:
        if run["method"] == "frozen":
            continue
        values = recomputed[run["run_id"]]
        retention.append({"run_id": run["run_id"], "method": run["method"], "seed": run["seed"], "epoch": run["epoch"],
                          "p1_delta": values["sugarcrepe_pp"]["positive1_accuracy"] - frozen["sugarcrepe_pp.positive1_accuracy"]["mean"],
                          "p2_delta": values["sugarcrepe_pp"]["positive2_accuracy"] - frozen["sugarcrepe_pp.positive2_accuracy"]["mean"],
                          "coco_t2i_r1_delta": values["coco_karpathy"]["t2i.r1"] - frozen["coco_karpathy.t2i.r1"]["mean"]})
    write_json(args.output / "retrieval_and_caption_retention.json", {"multipositive_vs_source_query_transitions": retrieval_transitions,
                                                                   "selected_adapters_vs_frozen": retention,
                                                                   "nonzero_selected_adapter_count": sum(row["epoch"] != 0 for row in retention),
                                                                   "status": "descriptive"})
    receipt = {"status": "passed", "source_index_sha256": sha256(args.index),
               "analysis_source_sha256": sha256(Path(__file__)), "run_count": len(index["runs"]),
               "dataset_count": len(DATASETS), "aggregate_checks": len(discrepancies),
               "maximum_raw_prediction_recomputation_difference": max(discrepancies),
               "bootstrap_replicates": args.replicates, "bootstrap_seed": args.bootstrap_seed,
               "historical_aggregates_used": False}
    write_json(args.output / "analysis_audit.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
