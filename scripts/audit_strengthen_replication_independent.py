#!/usr/bin/env python3
"""Independently reconstruct a replication's raw metrics and twelve effects."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np
from audit_allocation_distillation_independent import (
    Audit, ROOT, DATASETS, SEEDS, digest, read_json, canonical_digest,
    flatten_numeric, reconstruct_metrics, bootstrap_matrix, summarize_effect,
)

RATES = (1e-4, 3e-4, 1e-3)
POLICIES = ("source", "supported", "score_stratified_draw_0", "score_stratified_draw_1", "score_stratified_draw_2")
PROTOCOLS = {"nonlinear": ("3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920", 20261013),
             "rn50": ("53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e", 20261014)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--setting", choices=PROTOCOLS, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = Audit(ROOT)
    protocol_hash, seed = PROTOCOLS[args.setting]
    audit.check_hash(args.protocol, protocol_hash)
    protocol = read_json(audit.path(args.protocol))
    p = protocol["primary_analysis"]
    if (p["bootstrap_seed"] != seed or p["bootstrap_replicates"] != 10000 or p["family_size"] != 12 or
        p["epoch"] != 10 or p["learning_rates"] != list(RATES) or p["endpoints"] != ["i2t.r1", "t2i.r1"] or
        p["dataset"] != "e_vil_test1000" or p["alpha"] != .05):
        raise ValueError("Replication inference differs from its frozen design")
    index = read_json(audit.path(args.index))
    if (index["status"] != "complete" or index["setting"] != args.setting or
        index["protocol_sha256"] != protocol_hash or set(index["datasets"]) != set(DATASETS)):
        raise ValueError("A complete matching evaluation index is required")
    audit.check_hash(index["prescore_receipt"], index["prescore_receipt_sha256"])
    receipt = read_json(audit.path(index["prescore_receipt"]))
    for key in ("setting", "encoder", "protocol_sha256", "selections", "evidence_type"):
        if receipt[key] != index[key]:
            raise ValueError("Evaluation index differs from its prescore receipt")
    for name, expected in receipt["source_hashes"].items():
        audit.check_hash(name, expected)
    for name in ("state_manifest", "ledger", "dataset_config"):
        record = receipt[name]
        audit.check_hash(record["path"], record["sha256"])
    for record in receipt["selection_audit"]["candidate_completion_receipts"]:
        audit.check_hash(record["path"], record["sha256"])
    ledger = read_json(audit.path(receipt["ledger"]["path"]))
    if canonical_digest(ledger["identity"]) != ledger["ledger_sha256"] or ledger["ledger_sha256"] != index["ledger_sha256"]:
        raise ValueError("Replication ledger content hash differs")
    for name, expected in ledger["identity"]["source_sha256"].items():
        audit.check_hash(name, expected)
    config = read_json(audit.path(receipt["dataset_config"]["path"]))
    manifests = {}
    for dataset, records in config["datasets"].items():
        for kind, path in records.items():
            audit.check_hash(path, receipt["input_hashes"][dataset][f"{kind}_sha256"])
        manifests[dataset] = read_json(audit.path(records["manifest"]))
    expected_states = {state["state_id"]: state for state in receipt["states"]}
    states = {state["state_id"]: state for state in index["runs"]}
    if set(states) != set(expected_states) or len(states) != len(index["runs"]) or len(states) != index["run_count"]:
        raise ValueError("Replication evaluation states are incomplete or duplicated")
    terminal = {}
    predictions = {}
    metrics = {}
    for sid, state in states.items():
        base_state = {key: value for key, value in state.items() if key != "datasets"}
        if base_state != expected_states[sid] or set(state["datasets"]) != set(DATASETS):
            raise ValueError("A scored state differs from its immutable plan")
        if state["checkpoint"]:
            audit.check_hash(state["checkpoint"], state["checkpoint_sha256"])
        if state["epoch"] == 10:
            key = state["method"], state["learning_rate"], state["seed"]
            if key in terminal:
                raise ValueError("Duplicate terminal state")
            terminal[key] = state
        for dataset, artifact in state["datasets"].items():
            audit.check_hash(artifact["predictions"], artifact["predictions_sha256"])
            audit.check_hash(artifact["metadata"], artifact["metadata_sha256"])
            metadata = read_json(audit.path(artifact["metadata"]))
            expected_provenance = {"prescore_receipt_sha256": index["prescore_receipt_sha256"],
                "state": base_state, "dataset": dataset, "input_hashes": receipt["input_hashes"][dataset]}
            if (metadata["provenance"] != expected_provenance or metadata["metrics"] != artifact["metrics"] or
                metadata["predictions_sha256"] != artifact["predictions_sha256"]):
                raise ValueError("Prediction provenance differs from the prescore plan")
            with np.load(audit.path(artifact["predictions"]), allow_pickle=False) as archive:
                raw = {name: archive[name] for name in archive.files}
            audit.raw_identity(dataset, raw, manifests[dataset])
            rebuilt = flatten_numeric(reconstruct_metrics(dataset, raw))
            published = flatten_numeric(artifact["metrics"])
            if set(rebuilt) != set(published):
                raise ValueError("Raw aggregate field sets differ")
            for name, value in rebuilt.items():
                audit.equal(value, published[name], "raw_aggregate_endpoints")
            if dataset == "e_vil_test1000":
                if len(raw["image_ids"]) != 1000 or len(raw["text_ids"]) != 5000:
                    raise ValueError("The primary gallery has different sizes")
                if Counter(raw["text_source_image_ids"]) != Counter({iid: 5 for iid in raw["image_ids"]}):
                    raise ValueError("Primary text queries do not form five-caption image clusters")
                predictions[sid] = raw
            metrics[(sid, dataset)] = rebuilt
            audit.counts["prediction_archives"] += 1
    if set(terminal) != {(method, rate, seed) for method in POLICIES for rate in RATES for seed in SEEDS}:
        raise ValueError("The complete 45-state terminal grid is required")
    analysis = audit.path(args.analysis)
    preanalysis = read_json(analysis / "preanalysis_receipt.json")
    audit.check_hash(preanalysis["evaluation_index"]["path"], preanalysis["evaluation_index"]["sha256"])
    for name, expected in preanalysis["source_hashes"].items():
        audit.check_hash(name, expected)
    published_audit = read_json(analysis / "analysis_audit.json")
    audit.check_hash(analysis / "preanalysis_receipt.json", published_audit["preanalysis_receipt_sha256"])
    contrasts_document = read_json(analysis / "primary_contrasts.json")
    if contrasts_document["protocol_sha256"] != protocol_hash or contrasts_document["family_size"] != 12:
        raise ValueError("Published contrast family differs")
    contrasts = {row["effect_id"]: row for row in contrasts_document["contrasts"]}
    if len(contrasts) != 12:
        raise ValueError("Twelve unique published effects are required")
    effects = []
    for direction in ("i2t", "t2i"):
        reference = predictions[terminal[("source", RATES[0], SEEDS[0])]["state_id"]]
        ids = reference["image_ids" if direction == "i2t" else "text_ids"]
        clusters = reference["image_ids" if direction == "i2t" else "text_source_image_ids"]
        def values(method, rate):
            rows = []
            for training_seed in SEEDS:
                raw = predictions[terminal[(method, rate, training_seed)]["state_id"]]
                if not np.array_equal(raw["image_ids" if direction == "i2t" else "text_ids"], ids):
                    raise ValueError("Primary query identities differ between states")
                if not np.array_equal(raw["image_ids" if direction == "i2t" else "text_source_image_ids"], clusters):
                    raise ValueError("Primary image clusters differ between states")
                rows.append(np.asarray(raw[f"{direction}_ranks"] == 1, dtype=float))
            return np.stack(rows)
        current = []
        for rate in RATES:
            supported = values("supported", rate)
            for comparator, methods in (("source", ["source"]), ("score_stratified", list(POLICIES[2:]))):
                drawn = np.stack([values(method, rate) for method in methods])
                average = sum(drawn) / len(drawn)
                effect_id = f"supported_minus_{comparator}__lr_{rate:g}__{direction}_r1"
                current.append({"effect_id": effect_id, "delta": supported - average, "rate": rate,
                    "comparator": comparator, "methods": methods, "supported": supported,
                    "drawn": drawn, "clusters": clusters})
        draws = bootstrap_matrix([row["delta"] for row in current], clusters, replicates=10000, seed=seed)
        with np.load(analysis / "primary_bootstrap_samples.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(contrasts):
                raise ValueError("Saved bootstrap effect IDs differ")
            for column, row in enumerate(current):
                published = contrasts[row["effect_id"]]
                samples = draws[:, column]
                audit.equal(samples, saved[row["effect_id"]], "bootstrap_samples")
                summary = summarize_effect(row["delta"], samples, clusters, 12)
                summary["bootstrap_seed"] = seed
                for field, value in summary.items():
                    if isinstance(value, (int, float, list)):
                        audit.equal(value, published[field], "effect_statistics")
                    elif value != published[field]:
                        raise ValueError("Published effect metadata differs")
                audit.equal(100 * summary["difference"], published["difference_percentage_points"], "percentage_point_statistics")
                audit.equal([100 * summary["ci_lower"], 100 * summary["ci_upper"]], published["ci_percentage_points"], "percentage_point_statistics")
                audit.equal(row["supported"].mean(axis=1), published["supported_seed_values"], "seed_and_draw_statistics")
                audit.equal(row["drawn"].mean(axis=2), published["comparator_draw_by_seed_values"], "seed_and_draw_statistics")
                if (published["seed_ids"] != list(SEEDS) or published["comparator_draw_methods"] != row["methods"] or
                    published["left"] != "supported" or published["right"] != row["comparator"] or
                    published["learning_rate"] != row["rate"] or published["epoch"] != 10):
                    raise ValueError("Published paired effect uses a different schedule or draw set")
                effects.append({"effect_id": row["effect_id"], **summary})
        print(json.dumps({"direction": direction, "effects_checked": 6, "samples_checked": draws.size}), flush=True)
    result = {"schema_version": 1, "status": "passed", "setting": args.setting,
        "protocol_sha256": protocol_hash, "states": len(states), "primary_effects": len(effects),
        "bootstrap_replicates_per_effect": 10000, "bootstrap_seed": seed,
        "counts": dict(audit.counts), "maximum_absolute_differences": dict(audit.maxima), "effects": effects,
        "scope": "all saved query ranks, score-derived correctness, raw numeric aggregates, 12 terminal effects, and all 120000 saved bootstrap draws",
        "excluded_scope": "independent feature extraction and retrieval rank reconstruction from dot products",
        "independence": "cluster multiplicity matrix; explicit quantile interpolation; no production statistics imports",
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__),
            "scripts/audit_allocation_distillation_independent.py": digest(ROOT / "scripts/audit_allocation_distillation_independent.py")},
        "input_sha256": {str(audit.path(path).relative_to(ROOT)): digest(audit.path(path)) for path in
            [args.index, args.protocol, str(analysis / "preanalysis_receipt.json"),
             str(analysis / "primary_contrasts.json"), str(analysis / "primary_bootstrap_samples.npz")]},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    output = audit.path(args.output)
    if output.exists() and read_json(output) != result:
        raise ValueError("Existing replication audit receipt differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "setting": args.setting, "counts": dict(audit.counts),
                      "maximum_absolute_differences": dict(audit.maxima)}), flush=True)


if __name__ == "__main__":
    main()
