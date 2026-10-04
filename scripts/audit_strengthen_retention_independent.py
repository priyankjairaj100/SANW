#!/usr/bin/env python3
"""Independently audit the original retention-v3 raw evidence and all 116 effects.

The independent AD auditor supplies generic file and raw-aggregate checks.
No production evaluation or statistics helper is imported.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform

import numpy as np
from audit_allocation_distillation_independent import (
    Audit as BaseAudit, ROOT, ENCODERS, DATASETS, ENDPOINTS, SEEDS, REPLICATES,
    EPSILON, digest, read_json, canonical_digest, flatten_numeric, bootstrap_matrix,
    summarize_effect, reconstruct_metrics, strategy_arrays, gate_checks,
)

FAMILIES = ("source", "supported", "allocation", "distilled", "wise_ft", "image_source_only", "reverse_source_only")
POLICIES = ("source", "supported", "allocation_0.5", "allocation_0.8", "distilled_1", "distilled_4", "distilled_16", "image_source_only", "reverse_source_only")
RATES = (1e-4, 3e-4, 1e-3)
CELLS = ("source", "supported", "image_source_only", "reverse_source_only")
PROTOCOL_SHA256 = "5624342a20dd33b4f235264f57599a8bad0486d70fa587a6d9269371ea00b5d4"
PRIMARY_SEED = 20261005
FACTORIAL_SEED = 20261006


def factorial_plan(manifest):
    table = {}
    for state in manifest["states"]:
        if state["method"] not in POLICIES:
            continue
        key = (state["method"], state["learning_rate"], state["seed"], state["epoch"])
        if key in table:
            raise ValueError("Duplicate state schedule")
        table[key] = state
    actual = {key for key in table if key[0] in POLICIES}
    expected = {(policy, rate, seed, epoch) for policy in POLICIES for rate in RATES for seed in SEEDS for epoch in range(11)}
    if actual != expected:
        raise ValueError("The original retention grid is incomplete")
    result = []
    for cell in CELLS:
        for rate in RATES:
            for seed in SEEDS:
                state = table[(cell, rate, seed, 10)]
                if state["alpha"] != 1. or state["draw_id"] is not None:
                    raise ValueError("Factorial state must use full unscaled terminal weights")
                result.append({"state_id": state["state_id"], "cell": cell,
                    "learning_rate": rate, "seed": seed, "epoch": 10})
    return result


class Audit(BaseAudit):
    def load_index(self, index_path, dataset_config_path, lock, lock_hash, protocol_hash):
        index = read_json(self.path(index_path))
        if index["status"] != "complete" or index["protocol_sha256"] != protocol_hash:
            raise ValueError("Audit requires complete protocol-bound evaluation indices")
        encoder = index["encoder"]
        record = lock["encoders"][encoder]
        manifest_path = self.path(index["manifest"])
        self.check_hash(manifest_path, index["manifest_sha256"])
        if index["manifest_sha256"] != record["manifest_sha256"] or index["selection_lock_sha256"] != lock_hash:
            raise ValueError("Evaluation differs from the common selection lock")
        manifest = read_json(manifest_path)
        if not manifest["matched_controls_complete"] or manifest["test_outcomes_used_for_selection"]:
            raise ValueError("Selection or matched-control completion is invalid")
        self.check_hash(index["prescore_receipt"], index["prescore_receipt_sha256"])
        receipt = read_json(self.path(index["prescore_receipt"]))
        for key in ("protocol_sha256", "encoder", "manifest_sha256", "selection_lock_sha256",
                    "source_hashes", "selections", "factorial_selections", "evidence_type"):
            if index[key] != receipt[key]:
                raise ValueError(f"Index differs from prescore receipt: {key}")
        for name, expected in receipt["source_hashes"].items():
            self.check_hash(name, expected)
        ledger_path = manifest_path.parent / "ledger.json"
        self.check_hash(ledger_path, receipt["ledger_file_sha256"])
        ledger = read_json(ledger_path)
        if canonical_digest(ledger["identity"]) != ledger["ledger_sha256"] or ledger["ledger_sha256"] != manifest["ledger_sha256"]:
            raise ValueError("Training ledger identity differs")
        for name, expected in ledger["identity"]["source_sha256"].items():
            self.check_hash(name, expected)
        for section in ("inputs", "development_retrieval", "assignments"):
            for value in ledger["identity"][section].values():
                if isinstance(value, dict) and "path" in value and "sha256" in value:
                    self.check_hash(value["path"], value["sha256"])
        for label in ("primary", "sensitivity"):
            self.check_hash(manifest_path.parent / f"selection_{label}.json", manifest["selection_sha256"][label])
        if manifest["selection_sha256"] != record["selection_sha256"]:
            raise ValueError("Selections differ from the shared lock")
        self.check_hash(dataset_config_path, receipt["dataset_config_sha256"])
        config = read_json(self.path(dataset_config_path))
        if config["encoder"] != encoder or set(config["datasets"]) != set(DATASETS):
            raise ValueError("Invalid encoder dataset config")
        manifests = {}
        for dataset, paths in config["datasets"].items():
            for kind, path in paths.items():
                self.check_hash(path, receipt["input_hashes"][dataset][f"{kind}_sha256"])
            manifests[dataset] = read_json(self.path(paths["manifest"]))
        actual_states = [{key: value for key, value in run.items() if key != "datasets"} for run in index["runs"]]
        if actual_states != receipt["runs"]:
            raise ValueError("Scored states differ from the prescore plan")
        all_states = {state["state_id"]: state for state in manifest["states"]}
        if len(all_states) != len(manifest["states"]):
            raise ValueError("The training manifest contains duplicate states")
        ids = {row["state_id"] for row in manifest["selections"] + factorial_plan(manifest)}
        if ({run["state_id"] for run in index["runs"]} != ids | {"frozen"} or
            len(index["runs"]) != len(ids) + 1):
            raise ValueError("Scoring must include exactly the selected and decomposition states")
        if index["selections"] != manifest["selections"] or index["factorial_selections"] != factorial_plan(manifest):
            raise ValueError("Scoring selection plan differs from the manifest")
        predictions, metrics = {}, {}
        import torch
        for run in index["runs"]:
            state = {key: value for key, value in run.items() if key != "datasets"}
            if run["state_id"] != "frozen":
                if state != all_states[run["state_id"]]:
                    raise ValueError("Scored state differs from the development manifest")
                self.check_hash(run["checkpoint"], run["checkpoint_sha256"])
                checkpoint = torch.load(self.path(run["checkpoint"]), map_location="cpu", weights_only=True)
                for key in ("method", "seed", "epoch", "learning_rate"):
                    if checkpoint[key] != state[key]:
                        raise ValueError("Checkpoint header differs from its state")
                if checkpoint["protocol_sha256"] != protocol_hash or checkpoint["ledger_sha256"] != manifest["ledger_sha256"]:
                    raise ValueError("Checkpoint provenance differs")
                norm = np.sqrt(sum(np.square(weight.numpy().astype(np.float64)).sum()
                                   for weight in checkpoint["state_dict"].values()))
                self.equal(norm, state["update_norm"], "checkpoint_norms")
            if set(run["datasets"]) != set(DATASETS):
                raise ValueError("A scored state lacks a benchmark")
            for dataset, artifact in run["datasets"].items():
                self.check_hash(artifact["predictions"], artifact["predictions_sha256"])
                self.check_hash(artifact["metadata"], artifact["metadata_sha256"])
                metadata = read_json(self.path(artifact["metadata"]))
                if (metadata["provenance"] != {"run": state, "dataset": dataset,
                    "prescore_receipt_sha256": index["prescore_receipt_sha256"]} or
                    metadata["predictions_sha256"] != artifact["predictions_sha256"] or
                    metadata["metrics"] != artifact["metrics"]):
                    raise ValueError("Prediction provenance differs from the plan")
                with np.load(self.path(artifact["predictions"]), allow_pickle=False) as archive:
                    raw = {name: archive[name] for name in archive.files}
                self.raw_identity(dataset, raw, manifests[dataset])
                rebuilt = flatten_numeric(reconstruct_metrics(dataset, raw))
                published = flatten_numeric(artifact["metrics"])
                if set(rebuilt) != set(published):
                    raise ValueError("Raw aggregate field sets differ")
                for key, value in rebuilt.items():
                    self.equal(value, published[key], "raw_aggregate_endpoints")
                predictions[(run["state_id"], dataset)] = raw
                metrics[(run["state_id"], dataset)] = rebuilt
                self.counts["prediction_archives"] += 1
            self.counts["states"] += 1
        return index, predictions, metrics


def selected(index, family, tolerance=1.):
    states = {run["state_id"]: run for run in index["runs"]}
    if family == "frozen":
        return [states["frozen"]]
    entries = [row for row in index["selections"] if row["family"] == family and row["tolerance_pp"] == tolerance]
    expected = Counter((seed, draw) for seed in SEEDS for draw in range(3)) if family == "matched_distilled" else Counter((seed, None) for seed in SEEDS)
    runs = [states[row["state_id"]] for row in entries]
    if Counter((run["seed"], run["draw_id"]) for run in runs) != expected:
        raise ValueError("Selected strategy lacks the exact seed and draw grid")
    for entry, state in zip(entries, runs, strict=True):
        if entry["seed"] != state["seed"] or entry.get("draw_id") != state["draw_id"]:
            raise ValueError("Selection identity differs from state identity")
    return runs


def factorial_deltas(source, supported, image_source_only, reverse_source_only):
    s, u, i, r = source, supported, image_source_only, reverse_source_only
    return {"image_main_effect": (s - u + i - r) / 2,
            "reverse_main_effect": (s - u + r - i) / 2,
            "interaction": (s + u) - (i + r)}


def build_effects(index, predictions):
    encoder = index["encoder"]
    states = {row["state_id"]: row for row in index["runs"]}
    distilled = {row["seed"]: row for row in selected(index, "distilled")}
    for run in selected(index, "matched_distilled"):
        reference = distilled[run["seed"]]
        if any(run[key] != reference[key] for key in ("epoch", "learning_rate", "beta")):
            raise ValueError("Random controls differ from the primary distilled schedule")
    result = []
    comparisons = [(family, "frozen") for family in FAMILIES]
    comparisons += [("distilled", family) for family in ("allocation", "wise_ft", "matched_distilled")]
    for dataset, metric in ENDPOINTS:
        strategies = {family: strategy_arrays(selected(index, family), predictions, dataset, metric)
                      for family in FAMILIES + ("frozen", "matched_distilled")}
        for left, right in comparisons:
            a, b = strategies[left], strategies[right]
            if not np.array_equal(a[1], b[1]) or not np.array_equal(a[2], b[2]):
                raise ValueError("Primary contrasts have different query identities")
            result.append({"effect_id": f"{encoder}__{left}_minus_{right}__{dataset}__{metric}",
                "encoder": encoder, "left": left, "right": right, "dataset": dataset, "metric": metric,
                "family_size": 80, "bootstrap_seed": PRIMARY_SEED,
                "delta": a[0] - b[0], "clusters": a[2]})
    expected = Counter((cell, rate, seed, 10) for cell in CELLS for rate in RATES for seed in SEEDS)
    actual = Counter((row["cell"], row["learning_rate"], row["seed"], row["epoch"])
                     for row in index["factorial_selections"])
    if actual != expected:
        raise ValueError("The full fixed terminal factorial is required")
    for rate in RATES:
        cells = {cell: [] for cell in CELLS}
        for row in index["factorial_selections"]:
            if row["learning_rate"] != rate:
                continue
            run = states[row["state_id"]]
            if (run["method"] != row["cell"] or run["seed"] != row["seed"] or
                run["learning_rate"] != rate or run["epoch"] != 10 or
                run["alpha"] != 1. or run["draw_id"] is not None):
                raise ValueError("A factorial cell has a different state schedule")
            cells[row["cell"]].append(run)
        for dataset, metric in ENDPOINTS[:2]:
            arrays = {cell: strategy_arrays(runs, predictions, dataset, metric)
                      for cell, runs in cells.items()}
            reference = arrays["source"]
            for array in arrays.values():
                if not np.array_equal(reference[1], array[1]) or not np.array_equal(reference[2], array[2]):
                    raise ValueError("Factorial cells have different query identities")
            for name, delta in factorial_deltas(*(arrays[cell][0] for cell in CELLS)).items():
                result.append({"effect_id": f"{encoder}__{name}__lr_{rate:.8g}__{dataset}__{metric}",
                    "encoder": encoder, "effect": name, "learning_rate": rate, "epoch": 10,
                    "dataset": dataset, "metric": metric, "family_size": 36,
                    "bootstrap_seed": FACTORIAL_SEED, "delta": delta, "clusters": reference[2]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs=2, required=True)
    parser.add_argument("--dataset-configs", nargs=2, required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = Audit(ROOT)
    audit.check_hash(args.protocol, args.protocol_sha256)
    protocol = read_json(audit.path(args.protocol))
    specification = protocol["evaluation"]
    if (args.protocol_sha256 != PROTOCOL_SHA256 or specification["primary_family_size"] != 80 or
        specification["bootstrap_seed"] != PRIMARY_SEED or
        specification["mechanism_factorial"]["family_size"] != 36 or
        specification["mechanism_factorial"]["bootstrap_seed"] != FACTORIAL_SEED):
        raise ValueError("The unchanged retention-v3 inference contract is required")
    lock = read_json(audit.path(args.selection_lock))
    lock_hash = digest(audit.path(args.selection_lock))
    if lock["protocol_sha256"] != args.protocol_sha256 or set(lock["encoders"]) != set(ENCODERS):
        raise ValueError("Invalid common selection lock")
    configs = {read_json(audit.path(path))["encoder"]: path for path in args.dataset_configs}
    data, effects = {}, []
    for path in args.indices:
        encoder = read_json(audit.path(path))["encoder"]
        if encoder in data:
            raise ValueError("Duplicate encoder index")
        data[encoder] = audit.load_index(path, configs[encoder], lock, lock_hash, args.protocol_sha256)
        index, predictions, _ = data[encoder]
        effects.extend(build_effects(index, predictions))
        print(json.dumps({"raw_audit_complete": encoder, "states": len(index["runs"])}), flush=True)
    if set(data) != set(ENCODERS) or Counter(row["family_size"] for row in effects) != {80: 80, 36: 36}:
        raise ValueError("The independent effect grid is incomplete")
    analysis = audit.path(args.analysis)
    published = {}
    for name, count in (("primary_contrasts.json", 80), ("factorial_contrasts.json", 36)):
        document = read_json(analysis / name)
        if document["protocol_sha256"] != args.protocol_sha256 or document["family_size"] != count or len(document["contrasts"]) != count:
            raise ValueError("Published contrast family is incomplete")
        for row in document["contrasts"]:
            if row["effect_id"] in published:
                raise ValueError("Duplicate published effect")
            published[row["effect_id"]] = row
    if set(published) != {row["effect_id"] for row in effects}:
        raise ValueError("Published effects differ from the independently reconstructed grid")
    receipt = read_json(analysis / "preanalysis_receipt.json")
    if receipt["protocol_sha256"] != args.protocol_sha256:
        raise ValueError("Analysis receipt protocol differs")
    for name, expected in {**receipt["source_hashes"], **receipt["indices"]}.items():
        audit.check_hash(name, expected)
    production_audit = read_json(analysis / "analysis_audit.json")
    audit.check_hash(analysis / "preanalysis_receipt.json", production_audit["preanalysis_receipt_sha256"])
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row["dataset"], row["metric"], row["bootstrap_seed"], tuple(map(str, row["clusters"])))].append(row)
    reconstructed = {}
    with np.load(analysis / "bootstrap_samples.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(published):
            raise ValueError("Saved bootstrap arrays have missing or additional effects")
        for (dataset, metric, bootstrap_seed, _), group in grouped.items():
            draws = bootstrap_matrix([row["delta"] for row in group], group[0]["clusters"], seed=bootstrap_seed)
            for column, row in enumerate(group):
                key = row["effect_id"]
                samples = draws[:, column]
                audit.equal(samples, saved[key], "bootstrap_samples")
                summary = summarize_effect(row["delta"], samples, row["clusters"], row["family_size"])
                summary["bootstrap_seed"] = bootstrap_seed
                for field, value in summary.items():
                    if isinstance(value, (int, float, list)):
                        audit.equal(value, published[key][field], "effect_statistics")
                    elif value != published[key][field]:
                        raise ValueError("A published effect metadata field differs")
                for field in ("encoder", "dataset", "metric", "left", "right", "effect", "learning_rate", "epoch"):
                    if field in row and published[key][field] != row[field]:
                        raise ValueError("Published effect identity differs")
                reconstructed[key] = {**row, **summary}
            print(json.dumps({"bootstrap_audit_complete": dataset, "metric": metric,
                              "effects": len(group), "replicates": REPLICATES}), flush=True)
    gates_document = read_json(analysis / "practical_success_gates.json")
    gate_records = {(row["encoder"], row["family"]): row for row in gates_document["gates"]}
    if set(gate_records) != {(encoder, family) for encoder in ENCODERS for family in FAMILIES}:
        raise ValueError("Practical gate grid differs")
    for (encoder, family), reported in gate_records.items():
        rows = [row for row in reconstructed.values() if row["encoder"] == encoder and
                row.get("left") == family and row.get("right") == "frozen"]
        checks = gate_checks(selected(data[encoder][0], family), rows)
        if checks != reported["checks"] or all(checks.values()) != reported["passed"]:
            raise ValueError("A practical gate differs from independently reconstructed intervals")
        audit.counts["practical_gates"] += 1
    for row in gates_document["cross_encoder"]:
        if row["passed_both_encoders"] != all(gate_records[(encoder, row["family"])]["passed"] for encoder in ENCODERS):
            raise ValueError("Cross-encoder gate differs")
        audit.counts["cross_encoder_gates"] += 1
    if Counter(row["family"] for row in gates_document["cross_encoder"]) != Counter(FAMILIES):
        raise ValueError("Cross-encoder gate families are incomplete")
    # Check all descriptive reports directly against per-state raw aggregates.
    descriptions = read_json(analysis / "selected_strategy_metrics.json")["strategies"]
    seen = set()
    for description in descriptions:
        encoder, family, tolerance = (description[key] for key in ("encoder", "family", "tolerance_pp"))
        key = encoder, family, tolerance
        if key in seen:
            raise ValueError("Duplicate strategy description")
        seen.add(key)
        index, _, metrics = data[encoder]
        runs = selected(index, family, tolerance)
        if {r["state_id"] for r in description["states"]} != {r["state_id"] for r in runs}:
            raise ValueError("Strategy description references different states")
        for label, numbers in description["metrics"].items():
            dataset, metric = label.split(".", 1)
            seeds = (None,) if family == "frozen" else SEEDS
            means = [float(np.mean([metrics[(run["state_id"], dataset)][metric]
                     for run in runs if run["seed"] == seed])) for seed in seeds]
            audit.equal(means, numbers["seed_means"], "descriptive_statistics")
            audit.equal(np.mean(means), numbers["mean"], "descriptive_statistics")
            if len(means) > 1:
                audit.equal(np.std(means, ddof=1), numbers["seed_std"], "descriptive_statistics")
            elif numbers["seed_std"] is not None:
                raise ValueError("Frozen standard deviation must be null")
    expected_descriptions = {(encoder, family, tolerance) for encoder in ENCODERS for family in FAMILIES for tolerance in (0., 1.)}
    expected_descriptions |= {(encoder, "frozen", None) for encoder in ENCODERS}
    expected_descriptions |= {(encoder, "matched_distilled", 1.) for encoder in ENCODERS}
    if seen != expected_descriptions:
        raise ValueError("Descriptive strategies are incomplete")
    csv_seen = set()
    with (analysis / "all_state_metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["encoder"], row["state_id"], row["dataset"], row["metric"]
            if key in csv_seen:
                raise ValueError("Duplicate state metric row")
            csv_seen.add(key)
            audit.equal(float(row["value"]), data[row["encoder"]][2][(row["state_id"], row["dataset"])][row["metric"]], "csv_endpoints")
    expected_csv = set()
    for encoder, (index, _, metrics) in data.items():
        for (sid, dataset), values in metrics.items():
            for metric in values:
                if metric.endswith("accuracy") or metric.startswith(("i2t.", "t2i.")) or metric == "mean_bidirectional_r1":
                    expected_csv.add((encoder, sid, dataset, metric))
    if csv_seen != expected_csv:
        raise ValueError("CSV report lacks expected raw endpoint rows")
    output = audit.path(args.output)
    evidence_paths = args.indices + args.dataset_configs + [args.protocol, args.selection_lock]
    evidence_paths += [str(path.relative_to(ROOT)) for path in analysis.iterdir()
                       if path.is_file() and path.resolve() != output]
    result = {"schema_version": 1, "status": "passed", "protocol_sha256": args.protocol_sha256,
        "scope": "saved ranks, score-derived correctness, receipts, all 116 effects, all 100000 draws per effect, gates, and descriptive reports",
        "excluded_scope": "independent feature extraction and recomputation of retrieval ranks from feature dot products",
        "independent_implementation": "no project evaluator or statistics imports; cluster multiplicity matrix and explicit quantile interpolation",
        "primary_effects": 80, "factorial_effects": 36, "replicates_per_effect": REPLICATES,
        "primary_bootstrap_seed": PRIMARY_SEED, "factorial_bootstrap_seed": FACTORIAL_SEED, "counts": dict(audit.counts), "maximum_absolute_differences": dict(audit.maxima),
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__),
            "scripts/audit_allocation_distillation_independent.py": digest(ROOT / "scripts/audit_allocation_distillation_independent.py")},
        "input_sha256": {str(audit.path(path).relative_to(ROOT)): digest(audit.path(path)) for path in evidence_paths},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    if output.exists() and read_json(output) != result:
        raise ValueError("Existing independent audit receipt differs; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
