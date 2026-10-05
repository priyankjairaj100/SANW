#!/usr/bin/env python3
"""Select a shared finite-grid v9 family using exact paired inner gains.

The two families are selected independently after their own complete coverage.
Reported bootstrap intervals condition on the selected state and do not account
for this single inner selection. They are not confirmatory evidence.
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
from evaluate_practical_inner_v9 import verified_inner_run
from gcr.practical_inner_evaluation_v9 import (
    ENCODERS, exact_changes, paired_changes, select_shared, selected_uncertainty,
)


def archive(entry):
    with np.load(verify_record(entry), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def same_arrays(left, right, description):
    if set(left) != set(right) or any(not np.array_equal(left[key], right[key]) for key in left):
        raise ValueError(description)


def verify_index(filename, protocol_hash, family):
    index = read(filename)
    if index["study"] != "sanw_practical_v9_inner_index" or index["protocol_sha256"] != protocol_hash:
        raise ValueError("Wrong inner index identity")
    start = read(verify_record(index["start_receipt"]))
    completion_path = verify_record(start["completion"])
    verify_record(start["ledger"])
    identity, completion, protocol, split, expected_states = verified_inner_run(completion_path.parent, protocol_hash)
    if start["official_development_or_benchmarks_used"] is not False:
        raise ValueError("Index does not declare inner-only evaluation")
    if start["split"] != protocol["split"] or start["inputs"] != protocol["training_inputs"][identity["encoder"]]:
        raise ValueError("Index split/input identity mismatch")
    for source, expected in start["sources"].items():
        if digest(ROOT / source) != expected or protocol["source_sha256"][source] != expected:
            raise ValueError("Inner evaluator source changed")
    if index["encoder"] != identity["encoder"] or start["encoder"] != identity["encoder"]:
        raise ValueError("Index and training encoder disagree")
    if [(item["epoch"], item["family"]) for item in index["states"]] != [(item["epoch"], item["family"]) for item in expected_states]:
        raise ValueError("Index does not cover exactly the protocol-bound fit states")
    if any(item["family"] != family for item in index["states"]):
        raise ValueError("An index belongs to another family or descriptive control")
    output = []
    identity_arrays, identity_record = None, None
    for indexed, expected_state in zip(index["states"], expected_states):
        result_path = verify_record(indexed["result"])
        result = read(result_path)
        if result["start_receipt"] != index["start_receipt"] or result["protocol_sha256"] != protocol_hash:
            raise ValueError("State result is not bound to this evaluation index")
        if result["split"] != protocol["split"] or result["encoder"] != index["encoder"] or result["seed"] != 17:
            raise ValueError("State split/encoder/seed mismatch")
        for key, value in expected_state.items():
            if key != "scorer" and result.get(key) != value:
                raise ValueError(f"State metadata differs from its verified training checkpoint: {key}")
        baseline, trained = archive(result["frozen"]), archive(result["predictions"])
        changes, paired = paired_changes(baseline, trained)
        exact = exact_changes(paired)
        same_arrays(paired, archive(result["paired_predictions"]), "Saved paired arrays disagree with raw predictions")
        if changes != result["paired_changes"] or exact != result["exact_paired_changes"]:
            raise ValueError("Saved paired gains disagree with exact count reconstruction")
        if result["epoch"] == 0:
            if family != "labclip" or not result.get("identity_control"):
                raise ValueError("Unexpected zero-epoch candidate")
            identity_arrays, identity_record = trained, result["predictions"]
            continue
        row = {key: value for key, value in result.items() if key not in ("summary", "frozen_summary")}
        row["result"] = record(result_path)
        if family == "labclip":
            if identity_arrays is None or result["identity"] != identity_record:
                raise ValueError("Learned LABCLIP state lacks the correct identity-normalized comparator")
            identity_changes, identity_paired = paired_changes(identity_arrays, trained)
            same_arrays(identity_paired, archive(result["identity_paired_predictions"]), "Identity attribution arrays disagree")
            if identity_changes != result["identity_paired_changes"] or exact_changes(identity_paired) != result["exact_identity_paired_changes"]:
                raise ValueError("Learned-versus-identity gain disagrees with exact count reconstruction")
            parity_changes, parity_paired = paired_changes(baseline, identity_arrays)
            row["identity_vs_original_frozen"] = {"paired_changes": parity_changes, "exact_paired_changes": exact_changes(parity_paired)}
        output.append((row, baseline, trained, paired))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--family", choices=("joint", "labclip"), required=True)
    parser.add_argument("--indices", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = root_path(args.protocol)
    if digest(protocol_path) != args.protocol_sha256:
        raise ValueError("Selection protocol hash mismatch")
    protocol = read(protocol_path)
    if protocol["study"] != "sanw_practical_v9" or protocol["encoders"] != list(ENCODERS):
        raise ValueError("Wrong protocol study/encoder coverage")
    source_name = str(Path(__file__).resolve().relative_to(ROOT))
    if protocol["source_sha256"].get(source_name) != digest(__file__):
        raise ValueError("Selection CLI was not bound by the immutable protocol")
    verify_record(protocol["split"])
    if len({root_path(filename) for filename in args.indices}) != len(args.indices):
        raise ValueError("Duplicate evaluation index")
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a finite-grid selection")
    loaded = [item for filename in args.indices for item in verify_index(filename, args.protocol_sha256, args.family)]
    rows = [item[0] for item in loaded]
    # Verify an identical frozen comparator and all paired units within encoder.
    for encoder in ENCODERS:
        reference = None
        for row, frozen, _, _ in loaded:
            if row["encoder"] == encoder:
                if reference is not None:
                    same_arrays(reference, frozen, "Frozen comparator differs across finite-grid candidates")
                reference = frozen
    selected = select_shared(rows, family=args.family, radii=tuple(protocol["joint"]["radii"]),
                             weights=tuple(protocol["joint"]["composition_weights"]), epochs=tuple(protocol["labclip"]["epochs"]))
    output.mkdir(parents=True, exist_ok=True)
    start = {"study": "sanw_practical_v9_inner_selection_start", "family": args.family,
             "protocol": record(protocol_path), "split": protocol["split"], "indices": [record(p) for p in args.indices],
             "candidate_results": [row["result"] for row in rows], "source": record(__file__),
             "coverage_verified_before_selection": True, "candidate_count": len(rows),
             "exact_sign_and_tie_arithmetic": "Fraction from integer correct counts and per-image triplet counts",
             "official_development_or_benchmarks_used": False}
    start_record = write_json(output / "selection_start.json", start)
    result = {"study": "sanw_practical_v9_selection", "family": args.family, "passed": selected["passed"],
              "protocol_sha256": args.protocol_sha256, "split": protocol["split"], "start_receipt": start_record,
              "eligible_count": selected["eligible_count"], "candidate_count": len(rows),
              "candidates": rows, "selected_config": None, "selected_epoch": None,
              "selection_rule": "exact min encoder original gain; sum original gains; smaller radius; weight closer to 1 then smaller weight; or earlier baseline epoch",
              "eligibility": "both encoders: original gain strictly positive, other three gains nonnegative; LABCLIP original gain also strictly positive versus identity",
              "uncertainty_caveat": "one inner split and adaptive finite selection; reported intervals condition on selected states and omit selection uncertainty",
              "status": selected["status"], "official_development_or_benchmarks_used": False}
    if args.family == "labclip":
        result["arm"] = "small_data_fixed_scale"
    if selected["passed"]:
        if args.family == "joint":
            radius, weight = selected["selected_key"]
            result.update({"selected_radius": radius, "selected_composition_weight": weight,
                           "selected_config": {"radius": radius, "composition_weight": weight}})
        else:
            result["selected_epoch"] = selected["selected_key"][0]
        result["encoders"] = {}
        for state in selected["selected_states"]:
            row, _, _, paired = next(item for item in loaded if item[0]["result"] == state["result"])
            effects, samples = selected_uncertainty(paired)
            encoder = row["encoder"]
            entry = {"state": row, "effects": effects,
                     "bootstrap_samples": write_npz(output / f"{encoder}_bootstrap.npz", samples)}
            if args.family == "labclip":
                identity_effects, identity_samples = selected_uncertainty(archive(row["identity_paired_predictions"]))
                entry.update({"learned_vs_identity_effects": identity_effects,
                              "identity_bootstrap_samples": write_npz(output / f"{encoder}_vs_identity_bootstrap.npz", identity_samples),
                              "identity_vs_original_frozen": row["identity_vs_original_frozen"]})
            result["encoders"][encoder] = entry
    for entry in start["indices"]:
        verify_record(entry)
    selection_record = write_json(output / "selection.json", result)
    print(json.dumps({"selection": selection_record, "family": args.family, "passed": result["passed"],
                      "selected_config": result["selected_config"], "selected_epoch": result["selected_epoch"],
                      "eligible_count": result["eligible_count"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
