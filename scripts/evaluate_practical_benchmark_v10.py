#!/usr/bin/env python3
"""Lock, score, or analyze v10 practical benchmarks after full development qualification.

All twelve joint/control states, inputs, and evaluation sources are fixed before
the first benchmark score. No method selection, calibration, checkpoint search,
or fresh-confirmation access is implemented here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_practical_constrained_development_v8 import digest, read, record, root_path, verify_record, write_json, write_npz
import evaluate_practical_official_development_v10 as development
from aggregate_practical_official_development_v10 import aggregate_gate as development_gate, fixed_seed_pairs
from gcr.practical_inner_evaluation_v9 import exact_changes, paired_changes
from gcr.practical_exact_bootstrap_v10 import selected_development_uncertainty as selected_uncertainty
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer
from gcr.practical_benchmark_v10 import (SEEDS, ENCODERS, DATASETS, CONTRASTS, bootstrap, paired_seed_differences,
                                       practical_gate, score_retrieval, score_triplets)
from run_practical_streaming_v10 import verify_replication_gate
from evaluate_practical_v6 import load_dataset
from prepare_practical_benchmark_inputs_v10 import inventory, required_paths

FAMILIES = ("joint", "no_retention")
EXTRA_SOURCES = ("scripts/evaluate_practical_benchmark_v10.py", "src/gcr/practical_benchmark_v10.py",
                 "src/gcr/practical_exact_bootstrap_v10.py",
                 "scripts/prepare_practical_benchmark_inputs_v10.py", "scripts/evaluate_practical_v6.py",
                 "scripts/evaluate_practical_official_development_v10.py", "scripts/aggregate_practical_official_development_v10.py",
                 "scripts/evaluate_practical_official_development_v9.py", "scripts/aggregate_practical_official_development_v9.py")


def archive(entry):
    with np.load(verify_record(entry), allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def verify_development_gate(path, protocol_sha):
    """Reconstruct all six raw outcomes and the fixed-seed aggregate prerequisite."""
    gate = read(path)
    if (gate.get("study") != "sanw_practical_v10_fixed_seed_development_gate" or gate.get("family") != "joint"
            or gate.get("passed") is not True or gate.get("protocol_sha256") != protocol_sha):
        raise ValueError("The fixed-three-seed joint development gate must pass before benchmark access")
    start = read(verify_record(gate["start_receipt"]))
    verify_record(start["source"])
    if start["protocol"]["sha256"] != protocol_sha or start["fixed_seeds"] != list(SEEDS):
        raise ValueError("Development aggregate provenance differs")
    states, paired, baselines = {}, {}, {}
    locks = {}
    for entry in start["results"]:
        result = read(verify_record(entry))
        lock_entry = result["lock"]
        if lock_entry["sha256"] not in locks:
            locks[lock_entry["sha256"]] = development.load_lock(lock_entry["path"], lock_entry["sha256"])
        lock, state = locks[lock_entry["sha256"]], result["state"]
        key = state["encoder"], state["seed"]
        if (key in states or lock["protocol"]["sha256"] != protocol_sha or state != lock["states"][state["encoder"]]
                or result["checkpoint_sha256"] != state["checkpoint"]["sha256"]):
            raise ValueError("Development selected-state identity differs")
        own_gate = development.reconstruct_gate(result)
        if state["seed"] == 17 and not own_gate["passed"]:
            raise ValueError("Both initial development pilots must have passed")
        frozen, trained = archive(result["artifacts"]["frozen"]), archive(result["artifacts"]["trained"])
        _, paired[key] = paired_changes(frozen, trained)
        if state["encoder"] in baselines:
            old = baselines[state["encoder"]]
            if set(old) != set(frozen) or any(not np.array_equal(old[k], frozen[k]) for k in old):
                raise ValueError("Frozen development arrays changed between selected seeds")
        baselines[state["encoder"]] = frozen
        states[key] = state
    if set(states) != {(e, s) for e in ENCODERS for s in SEEDS}:
        raise ValueError("Six qualified development states required")
    for encoder in ENCODERS:
        combined = fixed_seed_pairs({seed: paired[encoder, seed] for seed in SEEDS})
        saved = archive(gate["artifacts"][f"{encoder}_paired"])
        if set(combined) != set(saved) or any(not np.array_equal(combined[k], saved[k]) for k in combined):
            raise ValueError("Aggregate development paired outcomes differ")
        exact = exact_changes(combined); effects, samples = selected_uncertainty(combined)
        saved_samples = archive(gate["artifacts"][f"{encoder}_bootstrap"])
        if any(not np.array_equal(samples[k], saved_samples[k]) for k in samples):
            raise ValueError("Aggregate development bootstrap differs")
        item = gate["encoders"][encoder]
        expected_states = [states[encoder, seed] for seed in SEEDS]
        if (item["states"] != expected_states or item["exact_paired_changes"] != exact
                or item["gate"] != development_gate(exact, effects, [state["nonzero"] for state in expected_states])
                or not item["gate"]["passed"]):
            raise ValueError("Fixed-seed development qualification differs from reconstructed evidence")
        for metric in effects:
            fields = ("difference", "ci_lower", "ci_upper", "replicates", "family_size", "bootstrap_seed")
            if metric in ("i2t", "t2i"):
                fields += ("exact_ci_lower", "exact_ci_upper")
            for field in fields:
                if effects[metric][field] != item["effects"][metric][field]:
                    raise ValueError("Reported development interval differs")
    return states


def verify_control(run, protocol, protocol_sha):
    run = root_path(run); ledger, completion = read(run / "ledger.json"), read(run / "completion.json")
    identity = ledger["identity"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != ledger["ledger_sha256"] or completion["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Control ledger changed")
    encoder, seed = identity["encoder"], identity["config"]["seed"]
    expected = {**protocol["fit_config"], "seed": seed}
    if (encoder not in ENCODERS or seed not in SEEDS
            or any(v.get("study") != "sanw_practical_v10" or v.get("family") != "no_retention" or v.get("mode") != "full"
                   or v.get("encoder") != encoder or v.get("config") != expected or v.get("protocol_sha256") != protocol_sha
                   for v in (identity, completion))
            or identity["source_sha256"] != protocol["source_sha256"]
            or identity["training_provenance"]["inputs"] != protocol["training_inputs"][encoder]
            or identity["training_provenance"]["original_training_inputs"] != protocol["original_training_inputs"][encoder]
            or identity["training_provenance"]["owner_sample"] != protocol["owner_sample"]
            or identity["training_provenance"]["confirmation_owner_lock"] != protocol["confirmation_owner_lock"]
            or identity["training_provenance"]["training_image_manifest_indices"] != list(range(6000))
            or identity["training_provenance"]["original_training_feature_bytes_identical"] is not True
            or identity["training_provenance"]["original_normalized_training_features_identical"] is not True
            or identity["training_provenance"]["heldout_used"] is not False
            or identity["fit_gallery_image_count"] != 6000 or identity["fit_gallery_text_count"] != 30000
            or identity["official_development_or_benchmarks_used"] is not False):
        raise ValueError("Matched control configuration or train-only scope changed")
    if digest(verify_record(identity["protocol"])) != protocol_sha:
        raise ValueError("Control protocol content differs")
    verify_replication_gate(ROOT, verify_record(identity["control_gate"]), protocol_sha)
    rows, history = completion["checkpoint_history"], completion["history"]
    steps_per_epoch = (6000 + expected["batch_size"] - 1) // expected["batch_size"]
    if ([row["epoch"] for row in history] != list(range(1, expected["epochs"] + 1))
            or any(not np.isfinite(row["training_objective"]) or row.get("retention_enforced") is not False for row in history)
            or any(row["optimizer_steps"] != row["epoch"] * steps_per_epoch for row in history)
            or completion["optimizer_steps"] != expected["epochs"] * steps_per_epoch
            or read(run / "history.json") != rows
            or [{k: v for k, v in row.items() if k != "checkpoint"} for row in rows] != history
            or completion.get("retention_enforced") is not False
            or completion.get("final_training_retention_diagnostic", {}).get("ranking_checked_canonically") is not True):
        raise ValueError("Control training budget or removed-retention semantics differ")
    for row in rows:
        entry = row["checkpoint"]
        if digest(run / entry["path"]) != entry["sha256"] or entry["ledger_sha256"] != ledger["ledger_sha256"]:
            raise ValueError("A control checkpoint or its ledger binding changed")
    chosen = min((row for row in history if row["nonzero"]), key=lambda row: (row["training_objective"], row["epoch"]))
    if (completion["selected_epoch"] != chosen["epoch"] or completion["selected_training_objective"] != chosen["training_objective"]
            or completion["selection"] != "minimum_nonzero_training_objective_then_earliest_epoch_without_feasibility_filter"):
        raise ValueError("Control was not selected by its fixed training objective")
    checkpoint = run / completion["selected_checkpoint"]["path"]
    epoch = run / rows[chosen["epoch"] - 1]["checkpoint"]["path"]
    if (digest(checkpoint) != completion["selected_checkpoint"]["sha256"]
            or digest(epoch) != rows[chosen["epoch"] - 1]["checkpoint"]["sha256"]):
        raise ValueError("Control checkpoint bytes changed")
    model, other = ConstrainedBilinearScorer.load(checkpoint), ConstrainedBilinearScorer.load(epoch)
    if any(not np.array_equal(getattr(model, key), getattr(other, key)) for key in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient")):
        raise ValueError("Control selected checkpoint differs from chosen epoch")
    norm2 = np.sum(model.coefficient * model.coefficient, dtype=np.float64)
    if not np.any(model.coefficient != 0) or not np.isfinite(norm2) or norm2 > expected["radius"] ** 2 * (1 + 128 * np.finfo(np.float64).eps):
        raise ValueError("Control must have a finite nonzero radius-constrained update")
    return {"run": str(run.relative_to(ROOT)), "encoder": encoder, "family": "no_retention", "seed": seed,
            "epoch": chosen["epoch"], "nonzero": True, "ledger": record(run / "ledger.json"),
            "completion": record(run / "completion.json"), "checkpoint": record(checkpoint)}


def verify_inputs(filename):
    inputs = read(filename); plan = inventory()
    if (inputs.get("study") != "sanw_practical_v10_benchmark_inputs" or inputs["scientific_index"] != plan["scientific_index"]
            or inputs["recovery_packages"] != plan["packages"]
            or inputs["restoration_source_sha256"] != digest(ROOT / "scripts/prepare_practical_benchmark_inputs_v10.py")):
        raise ValueError("Benchmark input recovery identity changed")
    expected = required_paths()
    if set(inputs["datasets"]) != set(expected):
        raise ValueError("Both benchmark encoders required")
    for encoder in ENCODERS:
        if set(inputs["datasets"][encoder]) != set(DATASETS):
            raise ValueError("All three prescribed datasets required")
        for dataset in DATASETS:
            entries = inputs["datasets"][encoder][dataset]
            if set(entries) != {"manifest", "features", "metadata"}:
                raise ValueError("Missing benchmark inputs")
            for key, entry in entries.items():
                if entry["path"] != expected[encoder][dataset][key]:
                    raise ValueError("Benchmark input path differs from recovered canonical cache")
                verify_record(entry)
    return inputs


def build_lock(protocol_path, protocol_sha, development_path, inputs_path, source_audit_path, runs):
    development.require_evaluation_threads()
    if digest(root_path(protocol_path)) != protocol_sha:
        raise ValueError("Training protocol hash mismatch")
    protocol = read(protocol_path)
    if protocol.get("study") != "sanw_practical_v10":
        raise ValueError("Wrong fitting study")
    expected_gate = {"bootstrap_replicates": 100000, "bootstrap_seed": 20261007,
                     "caption_endpoint": "SugarCrepe++_both_positives_strictly_above_negative",
                     "caption_improvement_adjusted_lower_strictly_greater_than": 0.0,
                     "encoders": list(ENCODERS), "family_size": 80, "familywise_alpha": .05,
                     "nonzero_trained_seeds": list(SEEDS), "primary_retrieval": "e_vil_test1000_full_gallery_R1_both_directions",
                     "resampling": "paired_image_clusters_fixed_selected_seed_mean",
                     "retrieval_adjusted_lower_strictly_greater_than": -.01, "secondary": "COCO_full_gallery_retrieval"}
    if protocol["practical_gate"] != expected_gate:
        raise ValueError("Practical gate differs from the recovered frozen contract")
    for source, sha in protocol["source_sha256"].items():
        if digest(ROOT / source) != sha:
            raise ValueError("A frozen fitting source changed")
    sources = {name: digest(ROOT / name) for name in EXTRA_SOURCES}
    audit = read(source_audit_path)
    if (audit.get("study") != "sanw_practical_v10_independent_benchmark_source_audit" or audit.get("passed") is not True
            or audit.get("blocking_findings") != [] or audit["protocol"]["sha256"] != protocol_sha
            or any(audit.get("sources", {}).get(name) != sha for name, sha in sources.items())):
        raise ValueError("Independent benchmark source review must bind every evaluator dependency")
    qualified = verify_development_gate(development_path, protocol_sha)
    states = {}
    for run in runs:
        family = read(root_path(run) / "ledger.json")["identity"]["family"]
        if family == "joint":
            _, _, state = development.verify_full_pilot(run, protocol, protocol_sha)
            if state != qualified[state["encoder"], state["seed"]]:
                raise ValueError("Benchmark candidate differs from the qualified development state")
        elif family == "no_retention":
            state = verify_control(run, protocol, protocol_sha)
        else:
            raise ValueError("Only the prespecified joint and no-retention families are eligible here")
        key = f"{state['encoder']}__{family}__{state['seed']}"
        if key in states:
            raise ValueError("Duplicate final state")
        states[key] = state
    if set(states) != {f"{e}__{f}__{s}" for e in ENCODERS for f in FAMILIES for s in SEEDS}:
        raise ValueError("Exactly twelve final candidate/control states required before any benchmark scoring")
    for encoder in ENCODERS:
        for seed in SEEDS:
            models = [ConstrainedBilinearScorer.load(verify_record(states[f"{encoder}__{family}__{seed}"]["checkpoint"])) for family in FAMILIES]
            if any(not np.array_equal(getattr(models[0], key), getattr(models[1], key)) for key in ("image_mean", "text_mean", "image_basis", "text_basis")):
                raise ValueError("Matched control training geometry differs")
    inputs = verify_inputs(inputs_path)
    return {"study": "sanw_practical_v10_benchmark_lock", "protocol": record(protocol_path),
            "development_gate": record(development_path), "inputs_receipt": record(inputs_path), "inputs": inputs["datasets"],
            "source_audit": record(source_audit_path), "extra_sources": sources, "states": states,
            "practical_gate": expected_gate, "fixed_contrasts": [value[0] for value in CONTRASTS], "effect_count": 30,
            "target_family": "joint", "secondary_coco_changes_do_not_change_primary_gate": True,
            "evaluation_status": protocol["evaluation_status"], "historical_numbers_used_as_baseline": False,
            "inference": "one canonical float64 centered bilinear pair score; same normalized frozen features for all states and tasks"}


def load_lock(path, sha):
    if digest(root_path(path)) != sha:
        raise ValueError("Benchmark lock SHA mismatch")
    lock = read(path)
    rebuilt = build_lock(lock["protocol"]["path"], lock["protocol"]["sha256"], lock["development_gate"]["path"],
                         lock["inputs_receipt"]["path"], lock["source_audit"]["path"], [state["run"] for state in lock["states"].values()])
    if lock != rebuilt:
        raise ValueError("Locked benchmark state, data, source, or prerequisite changed")
    return lock


def evaluate(args, lock):
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite benchmark outcomes")
    protocol = read(lock["protocol"]["path"])
    config = {"datasets": {name: {key: value["path"] for key, value in entries.items()}
                           for name, entries in lock["inputs"][args.encoder].items()}}
    metadata = read(verify_record(protocol["original_training_inputs"][args.encoder]["metadata"]))
    dataset = load_dataset(config, args.dataset, args.encoder, metadata)
    output.mkdir(parents=True, exist_ok=True)
    start = write_json(output / "prescore_receipt.json", {"lock": record(args.lock), "encoder": args.encoder, "dataset": args.dataset,
                       "normalization": "inherited single Torch float32 L2 pass; then float64 fixed pair reductions",
                       "historical_numbers_used_as_baseline": False, "no_fitting_or_selection": True})
    states = {"frozen": None} | {f"{family}_{seed}": lock["states"][f"{args.encoder}__{family}__{seed}"] for family in FAMILIES for seed in SEEDS}
    artifacts = {}
    for name, state in states.items():
        scorer = None
        if state is not None:
            model = ConstrainedBilinearScorer.load(verify_record(state["checkpoint"]))
            scorer = CanonicalScorer(model.image_mean, model.text_mean, model.image_basis, model.text_basis, model.coefficient)
        summary, raw = score_triplets(dataset, scorer) if args.dataset == "sugarcrepe_pp" else score_retrieval(dataset, scorer)
        artifacts[name] = {"state": state, "summary": summary, "predictions": write_npz(output / f"{name}.npz", raw)}
    # Final bytes check; models and benchmark inputs were locked before scoring.
    for entry in lock["inputs"][args.encoder][args.dataset].values():
        verify_record(entry)
    for state in states.values():
        if state:
            verify_record(state["checkpoint"])
    result = {"study": "sanw_practical_v10_benchmark_predictions", "status": "complete", "lock": record(args.lock),
              "encoder": args.encoder, "dataset": args.dataset, "start_receipt": start, "artifacts": artifacts}
    print(json.dumps(write_json(output / "index.json", result)), flush=True)


def analyze(args, lock):
    loaded, records = {}, []
    for filename in args.indices:
        index = read(filename); key = index["encoder"], index["dataset"]
        if (index.get("study") != "sanw_practical_v10_benchmark_predictions" or index.get("status") != "complete"
                or index["lock"]["sha256"] != args.lock_sha256 or key in loaded):
            raise ValueError("Benchmark index identity differs or is duplicated")
        verify_record(index["start_receipt"])
        expected = {"frozen"} | {f"{family}_{seed}" for family in FAMILIES for seed in SEEDS}
        if set(index["artifacts"]) != expected:
            raise ValueError("Incomplete fixed family/seed benchmark predictions")
        raw = {}
        for name, item in index["artifacts"].items():
            state = None if name == "frozen" else lock["states"][f"{key[0]}__{name.rsplit('_', 1)[0]}__{name.rsplit('_', 1)[1]}"]
            if item["state"] != state:
                raise ValueError("Benchmark predictions use an unlocked state")
            raw[name] = archive(item["predictions"])
        loaded[key] = raw; records.append(record(filename))
    if set(loaded) != {(e, d) for e in ENCODERS for d in DATASETS}:
        raise ValueError("All six encoder/dataset benchmark indices are required")
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite benchmark analysis")
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired").mkdir()
    effects, samples, paired_artifacts = [], {}, {}
    for (encoder, dataset), raw in loaded.items():
        metrics = ("both_accuracy",) if dataset == "sugarcrepe_pp" else ("i2t.r1", "t2i.r1")
        for contrast, first, second in CONTRASTS:
            for metric in metrics:
                by_seed = lambda family: {seed: raw["frozen" if family == "frozen" else f"{family}_{seed}"] for seed in SEEDS}
                delta, clusters = paired_seed_differences(dataset, metric, by_seed(first), by_seed(second))
                effect, values = bootstrap(delta, clusters)
                key = f"{encoder}__{dataset}__{metric.replace('.', '_')}__{contrast}"
                effects.append({**effect, "effect_id": key, "encoder": encoder, "dataset": dataset, "metric": metric, "contrast": contrast})
                samples[key] = values
                paired_artifacts[key] = write_npz(output / "paired" / f"{key}.npz", {"difference_by_seed": delta, "cluster_ids": clusters, "fixed_seeds": np.asarray(SEEDS)})
    if len(effects) != 30:
        raise ValueError("All thirty declared endpoint/contrast effects must be reported")
    result = {"study": "sanw_practical_v10_benchmark_analysis", "lock": record(args.lock), "indices": records,
              "effects": effects, "paired_artifacts": paired_artifacts, "bootstrap": write_npz(output / "bootstrap.npz", samples),
              "practical_gate": practical_gate(effects), "target_family": "joint", "no_test_selection": True,
              "evaluation_status": lock["evaluation_status"], "fresh_confirmation_used": False}
    print(json.dumps({"result": write_json(output / "analysis.json", result), "gate": result["practical_gate"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="mode", required=True)
    lock_parser = sub.add_parser("lock")
    for name in ("protocol", "protocol-sha256", "development-gate", "inputs", "source-audit", "output"):
        lock_parser.add_argument("--" + name, required=True)
    lock_parser.add_argument("--runs", nargs=12, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--encoder", choices=ENCODERS, required=True); evaluate_parser.add_argument("--dataset", choices=DATASETS, required=True)
    analysis_parser = sub.add_parser("analyze"); analysis_parser.add_argument("--indices", nargs=6, required=True)
    for command in (evaluate_parser, analysis_parser):
        for name in ("lock", "lock-sha256", "output"):
            command.add_argument("--" + name, required=True)
    args = parser.parse_args(); development.require_evaluation_threads()
    if args.mode == "lock":
        print(json.dumps(write_json(args.output, build_lock(args.protocol, args.protocol_sha256, args.development_gate,
                                                           args.inputs, args.source_audit, args.runs))), flush=True)
    else:
        lock = load_lock(args.lock, args.lock_sha256)
        (evaluate if args.mode == "evaluate" else analyze)(args, lock)


if __name__ == "__main__":
    main()
