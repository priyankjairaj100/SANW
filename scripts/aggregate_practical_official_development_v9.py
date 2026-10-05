#!/usr/bin/env python3
"""Fixed-three-seed official-development aggregation for a locked v9 family.

Replications need not pass individually. Their outcomes are averaged within
each paired query before image-cluster resampling. Seeds are never resampled.
Composition signs use exact rational arithmetic on integer triplet counts.
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
from evaluate_practical_official_development_v9 import load_lock, reconstruct_gate
from gcr.practical_inner_evaluation_v9 import METRICS, exact_changes, paired_changes, rational, selected_uncertainty


SEEDS = (17, 29, 43)


def archive(entry):
    with np.load(verify_record(entry), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def fixed_seed_pairs(paired_by_seed):
    """Exact per-item arithmetic, with all three selected states held fixed."""
    if set(paired_by_seed) != set(SEEDS):
        raise ValueError("Require exactly selected seeds 17,29,43")
    output = {"selected_seeds": np.asarray(SEEDS, dtype=np.int64)}
    reference = paired_by_seed[17]
    for metric in METRICS:
        denominator = reference[f"{metric}_difference_denominators"]
        ids = reference[f"{metric}_cluster_ids"]
        for seed in SEEDS:
            value = paired_by_seed[seed]
            if (not np.array_equal(value[f"{metric}_cluster_ids"], ids)
                    or not np.array_equal(value[f"{metric}_difference_denominators"], denominator)):
                raise ValueError("Fixed seeds do not share identical paired queries and triplet denominators")
        numerators = np.stack([paired_by_seed[seed][f"{metric}_difference_numerators"] for seed in SEEDS])
        summed = numerators.sum(axis=0, dtype=np.int64)
        scaled_denominator = len(SEEDS) * denominator
        output[f"{metric}_cluster_ids"] = ids
        output[f"{metric}_difference_numerators"] = summed
        output[f"{metric}_difference_denominators"] = scaled_denominator
        output[f"{metric}_difference"] = summed.astype(np.float64) / scaled_denominator
        output[f"{metric}_difference_numerators_by_seed"] = numerators
        output[f"{metric}_single_seed_denominators"] = denominator
    return output


def aggregate_gate(exact, effects, norms, identity_exact=None):
    if len(norms) != 3 or any(not np.isfinite(x) or x <= 0 for x in norms):
        raise ValueError("All three selected states require finite nonzero updates")
    checks = {"three_nonzero_trained_states": True, "original_joint_improvement": rational(exact["original"]) > 0,
              "source_pair_joint_non_decrease": rational(exact["source_pair"]) >= 0}
    for direction in ("i2t", "t2i"):
        checks[f"{direction}_mean_non_decrease"] = rational(exact[direction]) >= 0
        checks[f"{direction}_strict_adjusted_retention"] = effects[direction]["ci_lower"] > -.01
    if identity_exact is not None:
        checks["learned_original_joint_improvement_over_identity"] = rational(identity_exact["original"]) > 0
    return {"passed": all(checks.values()), "checks": checks, "individual_replication_pass_required": False,
            "fixed_seeds": list(SEEDS), "seed_resampling": False, "score_ensemble": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--results", nargs=6, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if digest(root_path(args.protocol)) != args.protocol_sha256:
        raise ValueError("Protocol hash mismatch")
    protocol = read(args.protocol)
    if (protocol["study"] != "sanw_practical_v9"
            or protocol["continuation"]["replication_rule"] != "same_fixed_3seed_mean_development_contract_before_benchmark_lock"):
        raise ValueError("Protocol does not declare this fixed-three-seed development rule")
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite fixed-seed development aggregation")
    loaded, family, selection_sha = {}, None, None
    for filename in args.results:
        result = read(filename)
        if result["study"] != "sanw_practical_v9_official_development_result":
            raise ValueError("Wrong single-state development result schema")
        lock = load_lock(result["lock"]["path"], result["lock"]["sha256"])
        if lock["protocol"]["sha256"] != args.protocol_sha256 or result["state"] != lock["states"][result["encoder"]]:
            raise ValueError("Single-state result differs from its protocol/state lock")
        if lock["extra_sources"].get(str(Path(__file__).resolve().relative_to(ROOT))) != digest(__file__):
            raise ValueError("This aggregator source was not bound before official-development scoring")
        family = family or lock["family"]
        selection_sha = selection_sha or lock["selection"]["sha256"]
        if lock["family"] != family or lock["selection"]["sha256"] != selection_sha:
            raise ValueError("A shared family and inner-selected configuration are required for all seeds")
        state = result["state"]
        key = (state["encoder"], state["seed"])
        if key in loaded or key[1] not in SEEDS:
            raise ValueError("Duplicate or unexpected selected seed")
        reconstructed_gate = reconstruct_gate(result)
        if state["seed"] == 17 and not reconstructed_gate["passed"]:
            raise ValueError("Both seed17 pilots must pass before replication aggregation")
        arrays = {name: archive(result["artifacts"][name]) for name in ("frozen", "trained")}
        _, paired = paired_changes(arrays["frozen"], arrays["trained"])
        identity_paired = None
        if family == "labclip":
            arrays["identity"] = archive(result["artifacts"]["identity"])
            _, identity_paired = paired_changes(arrays["identity"], arrays["trained"])
        loaded[key] = {"result": record(filename), "state": state, "arrays": arrays,
                       "paired": paired, "identity_paired": identity_paired, "lock": result["lock"]}
    if set(loaded) != {(encoder, seed) for encoder in ("vit_b32", "rn50") for seed in SEEDS}:
        raise ValueError("Require all six selected encoder/seed states")
    output.mkdir(parents=True, exist_ok=True)
    start = {"study": "sanw_practical_v9_fixed_seed_development_start", "protocol": record(args.protocol),
             "family": family, "selection_sha256": selection_sha, "source": record(__file__),
             "results": [value["result"] for value in loaded.values()],
             "state_locks": [value["lock"] for value in loaded.values()],
             "fixed_seeds": list(SEEDS), "individual_replication_pass_required": False,
             "no_new_benchmark_access": True}
    start_record = write_json(output / "aggregation_start.json", start)
    encoders, artifacts = {}, {}
    for encoder in ("vit_b32", "rn50"):
        values = [loaded[encoder, seed] for seed in SEEDS]
        reference = values[0]["arrays"]["frozen"]
        for value in values:
            for metric in METRICS:
                for suffix in ("correct", "cluster_ids"):
                    key = f"{metric}_{suffix}"
                    if not np.array_equal(reference[key], value["arrays"]["frozen"][key]):
                        raise ValueError("Frozen baseline or paired owner units changed between seeds")
        combined = fixed_seed_pairs({seed: loaded[encoder, seed]["paired"] for seed in SEEDS})
        exact = exact_changes(combined)
        effects, samples = selected_uncertainty(combined)
        for effect in effects.values():
            effect["conditioning"] = "fixed selected-seed mean within query; paired official-development image clusters resampled; seeds never resampled"
        artifacts[f"{encoder}_paired"] = write_npz(output / f"{encoder}_fixed_seed_paired.npz", combined)
        artifacts[f"{encoder}_bootstrap"] = write_npz(output / f"{encoder}_bootstrap.npz", samples)
        identity_exact, identity_effects = None, None
        if family == "labclip":
            combined_identity = fixed_seed_pairs({seed: loaded[encoder, seed]["identity_paired"] for seed in SEEDS})
            identity_exact = exact_changes(combined_identity)
            identity_effects, identity_samples = selected_uncertainty(combined_identity)
            artifacts[f"{encoder}_vs_identity_paired"] = write_npz(output / f"{encoder}_vs_identity_paired.npz", combined_identity)
            artifacts[f"{encoder}_vs_identity_bootstrap"] = write_npz(output / f"{encoder}_vs_identity_bootstrap.npz", identity_samples)
        encoders[encoder] = {"states": [value["state"] for value in values], "exact_paired_changes": exact,
                             "effects": effects, "learned_vs_identity_effects": identity_effects,
                             "exact_learned_vs_identity": identity_exact,
                             "gate": aggregate_gate(exact, effects, [value["state"]["update_norm"] for value in values], identity_exact)}
    for item in start["results"]:
        verify_record(item)
    final = {"study": "sanw_practical_v9_fixed_seed_development_gate", "family": family,
             "protocol_sha256": args.protocol_sha256, "selection_sha256": selection_sha,
             "start_receipt": start_record, "encoders": encoders, "artifacts": artifacts,
             "passed": all(value["gate"]["passed"] for value in encoders.values()),
             "status": "development_only_requires_separate_benchmark_lock_and_audit",
             "evaluation_status": protocol["evaluation_status"], "no_new_benchmark_access": True}
    print(json.dumps({"result": write_json(output / "result.json", final), "passed": final["passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
