#!/usr/bin/env python3
"""Aggregate the fixed v10 seeds per paired item before image-cluster resampling.

The 17/29/43 mean is fixed; seeds are never resampled. Individual replication
results need not pass the gate. The seed-17 pilot prerequisite remains required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_practical_constrained_development_v8 import digest, read, record, root_path, verify_record, write_json, write_npz
from evaluate_practical_official_development_v10 import gate_checks, load_lock, reconstruct_gate, require_evaluation_threads
from aggregate_practical_official_development_v9 import fixed_seed_pairs
from gcr.practical_inner_evaluation_v9 import METRICS, exact_changes, paired_changes
from gcr.practical_exact_bootstrap_v10 import selected_development_uncertainty as selected_uncertainty


SEEDS = (17, 29, 43)


def archive(entry):
    with np.load(verify_record(entry), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def aggregate_gate(exact, effects, nonzero):
    if len(nonzero) != 3 or any(value is not True for value in nonzero):
        raise ValueError("All three selected trained updates must be nonzero")
    single = gate_checks(exact, effects, True)
    single["checks"]["three_nonzero_trained_states"] = single["checks"].pop("nonzero_trained_update")
    return {**single, "individual_replication_pass_required": False, "fixed_seeds": list(SEEDS),
            "seed_resampling": False, "score_ensemble": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--results", nargs=6, required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_evaluation_threads()
    if digest(root_path(args.protocol)) != args.protocol_sha256:
        raise ValueError("Protocol hash mismatch")
    protocol = read(args.protocol)
    if (protocol.get("study") != "sanw_practical_v10" or protocol.get("seeds") != list(SEEDS)
            or protocol["continuation"]["replication_rule"] != "same_fixed_3seed_mean_development_contract_before_benchmark_lock"):
        raise ValueError("Protocol does not declare the prescribed fixed-three-seed route")
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite fixed-seed development evidence")
    loaded = {}
    for filename in args.results:
        result = read(filename)
        if result.get("study") != "sanw_practical_v10_official_development":
            raise ValueError("Wrong single-state result schema")
        lock = load_lock(result["lock"]["path"], result["lock"]["sha256"])
        state = result["state"]
        if (lock["protocol"]["sha256"] != args.protocol_sha256 or lock["family"] != "joint"
                or state != lock["states"][result["encoder"]] or result["seed"] != state["seed"]
                or result["protocol_sha256"] != args.protocol_sha256
                or result["checkpoint_sha256"] != state["checkpoint"]["sha256"]):
            raise ValueError("Single-state protocol or model identity differs from its lock")
        if lock["extra_sources"].get(str(Path(__file__).resolve().relative_to(ROOT))) != digest(__file__):
            raise ValueError("Aggregator source was not bound before official-development scoring")
        key = (state["encoder"], state["seed"])
        if key in loaded or key[1] not in SEEDS:
            raise ValueError("Duplicate or unexpected seed")
        gate = reconstruct_gate(result)
        if state["seed"] == 17 and not gate["passed"]:
            raise ValueError("Both seed-17 pilots must pass before replication aggregation")
        arrays = {name: archive(result["artifacts"][name]) for name in ("frozen", "trained")}
        _, paired = paired_changes(arrays["frozen"], arrays["trained"])
        loaded[key] = {"result": record(filename), "state": state, "arrays": arrays,
                       "paired": paired, "lock": result["lock"]}
    if set(loaded) != {(encoder, seed) for encoder in ("vit_b32", "rn50") for seed in SEEDS}:
        raise ValueError("Exactly six selected encoder/seed states required")
    output.mkdir(parents=True, exist_ok=True)
    start = {"study": "sanw_practical_v10_fixed_seed_development_start", "protocol": record(args.protocol),
             "family": "joint", "source": record(__file__), "results": [value["result"] for value in loaded.values()],
             "state_locks": [value["lock"] for value in loaded.values()], "fixed_seeds": list(SEEDS),
             "individual_replication_pass_required": False, "no_new_benchmark_access": True}
    start_record = write_json(output / "aggregation_start.json", start)
    encoders, artifacts = {}, {}
    for encoder in ("vit_b32", "rn50"):
        values = [loaded[encoder, seed] for seed in SEEDS]
        reference = values[0]["arrays"]["frozen"]
        for value in values:
            other = value["arrays"]["frozen"]
            keys = [f"{metric}_{suffix}" for metric in METRICS for suffix in ("correct", "cluster_ids")]
            keys += ["original_triplet_counts", "source_pair_triplet_counts", "gallery_image_ids", "gallery_text_ids"]
            if any(not np.array_equal(reference[key], other[key]) for key in keys):
                raise ValueError("Frozen baseline, gallery, or paired units changed across seeds")
        combined = fixed_seed_pairs({seed: loaded[encoder, seed]["paired"] for seed in SEEDS})
        exact = exact_changes(combined); effects, samples = selected_uncertainty(combined)
        for effect in effects.values():
            effect["conditioning"] = "fixed selected-seed mean per query; paired official-development image clusters resampled; seeds never resampled"
        artifacts[f"{encoder}_paired"] = write_npz(output / f"{encoder}_fixed_seed_paired.npz", combined)
        artifacts[f"{encoder}_bootstrap"] = write_npz(output / f"{encoder}_bootstrap.npz", samples)
        encoders[encoder] = {"states": [value["state"] for value in values], "exact_paired_changes": exact, "effects": effects,
                             "gate": aggregate_gate(exact, effects, [value["state"]["nonzero"] for value in values])}
    for item in start["results"]:
        verify_record(item)
    final = {"study": "sanw_practical_v10_fixed_seed_development_gate", "family": "joint",
             "protocol_sha256": args.protocol_sha256, "start_receipt": start_record, "encoders": encoders,
             "artifacts": artifacts, "passed": all(value["gate"]["passed"] for value in encoders.values()),
             "status": "development_only_requires_separate_benchmark_lock_and_audit",
             "evaluation_status": protocol["evaluation_status"], "no_new_benchmark_access": True}
    print(json.dumps({"result": write_json(output / "result.json", final), "passed": final["passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
