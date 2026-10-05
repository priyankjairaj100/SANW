#!/usr/bin/env python3
"""Separately locked official-development evaluation of qualified v9 pilots.

The original fitting protocol remains unchanged. This new source must be bound
in a two-encoder evaluation lock before either official-development result is
computed. No benchmark loader exists in this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_practical_constrained_development_v8 import (
    CONTRACT, digest, load_development, read, record, root_path, verify_record, write_json, write_npz,
)
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer, composition_metrics, exact_retrieval
from gcr.practical_inner_evaluation_v9 import LABCLIPScorer, exact_changes, paired_changes, rational, selected_uncertainty


def verify_full_pilot(run, protocol, protocol_hash, selection_path):
    run = root_path(run)
    ledger, completion = read(run / "ledger.json"), read(run / "completion.json")
    identity = ledger["identity"]
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != ledger["ledger_sha256"] or completion["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Full pilot ledger identity mismatch")
    seed = identity["config"]["seed"]
    if identity["mode"] != "full" or completion["mode"] != "full" or seed not in (17, 29, 43):
        raise ValueError("This lock accepts only independently qualified declared full-data seeds")
    if identity["encoder"] not in protocol["encoders"] or completion["encoder"] != identity["encoder"]:
        raise ValueError("Full pilot encoder mismatch")
    if digest(verify_record(identity["protocol"])) != protocol_hash or identity["split"] != protocol["split"]:
        raise ValueError("Full pilot protocol/split mismatch")
    selection = read(selection_path)
    if (selection["study"] != "sanw_practical_v9_selection" or selection["passed"] is not True
            or selection["protocol_sha256"] != protocol_hash or selection["split"] != protocol["split"]):
        raise ValueError("A passing shared inner selection is required before full-pilot evaluation")
    if digest(verify_record(identity["selection"])) != digest(selection_path):
        raise ValueError("Pilot was not fit using the declared shared inner selection")
    if seed != 17:
        gate_record = identity.get("replication_gate") or identity["selection"].get("replication_gate")
        if not gate_record:
            raise ValueError("Replication fit does not bind a passing seed17 development gate")
        gate = read(verify_record(gate_record))
        if (gate.get("study") != "sanw_practical_v9_replication_gate" or gate.get("passed") is not True
                or gate.get("family") != selection["family"] or gate.get("protocol_sha256") != protocol_hash
                or gate.get("selection_sha256") != digest(selection_path)
                or gate.get("seed") != 17 or gate.get("encoders_passed") != ["vit_b32", "rn50"]):
            raise ValueError("Replication authorization differs from this selected family and protocol")
    inputs = identity["training_provenance"]["inputs"]
    if inputs != protocol["training_inputs"][identity["encoder"]]:
        raise ValueError("Full fitting used different training features")
    for item in inputs.values():
        verify_record(item)
    for source, expected in identity["source_sha256"].items():
        if protocol["source_sha256"][source] != expected or digest(ROOT / source) != expected:
            raise ValueError("Full fitting source changed")
    split = read(verify_record(protocol["split"]))
    all_training = sorted(split["train_image_manifest_indices"] + split["validation_image_manifest_indices"])
    if identity["training_provenance"]["training_image_manifest_indices"] != all_training:
        raise ValueError("Full pilot did not use the original 1200-owner training pool")
    if completion["config"] != identity["config"]:
        raise ValueError("Full pilot completion configuration mismatch")
    config, family = identity["config"], selection["family"]
    if family == "joint":
        if identity["study"] != "sanw_practical_v9" or identity["family"] != "joint":
            raise ValueError("Wrong full joint study")
        expected = {**protocol["joint"]["base_config"], **selection["selected_config"], "seed": seed}
        if config != expected:
            raise ValueError("Full joint configuration differs from the common inner selection")
        history = completion["history"]
        if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError("Incomplete full joint training budget")
        if any(not np.isfinite(row["training_objective"]) for row in history):
            raise ValueError("Nonfinite full joint objective")
        chosen = min((row for row in history if row["nonzero"]), key=lambda row: (row["training_objective"], row["epoch"]))
        if chosen["epoch"] != completion["selected_epoch"] or chosen["training_objective"] != completion["selected_training_objective"]:
            raise ValueError("Full joint state was not training-objective-selected")
        certificate = completion["final_certificate"]
        if not all(certificate.get(key) is True for key in ("feasible_with_tolerance", "ranking_preserved", "ranking_checked_canonically")):
            raise ValueError("Missing full joint feasibility certificate")
        checkpoint = run / completion["selected_checkpoint"]["path"]
        if digest(checkpoint) != completion["selected_checkpoint"]["sha256"]:
            raise ValueError("Full joint checkpoint changed")
        model = ConstrainedBilinearScorer.load(checkpoint)
        epoch_row = completion["checkpoint_history"][chosen["epoch"] - 1]
        epoch_checkpoint = run / epoch_row["checkpoint"]["path"]
        if digest(epoch_checkpoint) != epoch_row["checkpoint"]["sha256"]:
            raise ValueError("Full joint selected epoch changed")
        same_epoch = ConstrainedBilinearScorer.load(epoch_checkpoint)
        if any(not np.array_equal(getattr(model, key), getattr(same_epoch, key))
               for key in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient")):
            raise ValueError("Full selected joint model differs from its selected epoch")
        norm = float(np.linalg.norm(model.coefficient))
        if not 0 < norm <= config["radius"] * (1 + 64 * np.finfo(np.float64).eps):
            raise ValueError("Invalid full joint update norm")
        scorer = CanonicalScorer(model.image_mean, model.text_mean, model.image_basis, model.text_basis, model.coefficient)
        epoch, identity_scorer = chosen["epoch"], None
    elif family == "labclip":
        if identity["study"] != "sanw_labclip_inner_v9" or config["arm"] != "small_data_fixed_scale":
            raise ValueError("Wrong selected LABCLIP study/arm")
        if config != {**protocol["labclip"]["arms"]["small_data_fixed_scale"]["config"], "seed": seed}:
            raise ValueError("Full LABCLIP recipe changed")
        epoch = selection["selected_epoch"]
        if identity["stop_epoch"] != epoch or [row["epoch"] for row in completion["history"]] != list(range(1, epoch + 1)):
            raise ValueError("LABCLIP full fit did not stop at the inner-selected epoch")
        requested = identity["requested_fit_loader_owner_indices"]
        if sorted(requested) != list(range(1200)):
            raise ValueError("Full LABCLIP fit did not request all original training owners")
        row = next((row for row in completion["checkpoint_history"] if row["epoch"] == epoch), None)
        if row is None or row["nonzero_functional_update"] is not True:
            raise ValueError("Selected LABCLIP checkpoint lacks a functional learned update")
        checkpoint = run / row["checkpoint"]["path"]
        if digest(checkpoint) != row["checkpoint"]["sha256"]:
            raise ValueError("Full LABCLIP checkpoint changed")
        with np.load(checkpoint, allow_pickle=False) as archive:
            if str(archive["schema"]) != "sanw_labclip_inner_v9":
                raise ValueError("Wrong LABCLIP checkpoint schema")
            weight = archive["weight"]
        scorer, identity_scorer = LABCLIPScorer(weight), LABCLIPScorer(np.eye(len(weight)))
        norm = float(np.linalg.norm(weight - np.eye(len(weight))))
    else:
        raise ValueError("Only selected joint or adapted LABCLIP families qualify")
    return identity, scorer, identity_scorer, {"run": str(run.relative_to(ROOT)), "encoder": identity["encoder"],
        "family": family, "seed": seed, "epoch": epoch, "update_norm": norm,
        "ledger": record(run / "ledger.json"), "completion": record(run / "completion.json"), "checkpoint": record(checkpoint)}


def build_lock(protocol_path, protocol_hash, selection_path, runs):
    protocol_path, selection_path = root_path(protocol_path), root_path(selection_path)
    if digest(protocol_path) != protocol_hash:
        raise ValueError("Protocol hash mismatch")
    protocol = read(protocol_path)
    expected_contract = {**CONTRACT, "selection": "one_preselected_fullfit_state_no_development_search"}
    if protocol["study"] != "sanw_practical_v9" or protocol["development_contract"] != expected_contract:
        raise ValueError("Wrong official-development contract")
    for source, expected in protocol["source_sha256"].items():
        if digest(ROOT / source) != expected:
            raise ValueError("A fitting or inner-selection source changed")
    states = {}
    for run in runs:
        _, _, _, state = verify_full_pilot(run, protocol, protocol_hash, selection_path)
        if state["encoder"] in states:
            raise ValueError("Duplicate full-pilot encoder")
        states[state["encoder"]] = state
    if (set(states) != {"vit_b32", "rn50"} or len({state["family"] for state in states.values()}) != 1
            or len({state["seed"] for state in states.values()}) != 1):
        raise ValueError("One common family, one fixed seed, and both full-fit encoders required")
    for entries in protocol["development_inputs"].values():
        for item in entries.values():
            verify_record(item)
    return {"study": "sanw_practical_v9_official_development_lock", "protocol": record(protocol_path),
            "selection": record(selection_path), "family": next(iter(states.values()))["family"],
            "seed": next(iter(states.values()))["seed"],
            "states": states, "inputs": protocol["development_inputs"], "contract": expected_contract,
            "extra_sources": {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__),
                              "scripts/aggregate_practical_official_development_v9.py": digest(ROOT / "scripts/aggregate_practical_official_development_v9.py")},
            "evaluation_status": protocol["evaluation_status"], "no_new_benchmark_access": True}


def load_lock(path, expected):
    if digest(root_path(path)) != expected:
        raise ValueError("Official-development lock hash mismatch")
    lock = read(path)
    rebuilt = build_lock(lock["protocol"]["path"], lock["protocol"]["sha256"], lock["selection"]["path"],
                         [state["run"] for state in lock["states"].values()])
    if lock != rebuilt:
        raise ValueError("Official-development sources or locked states changed")
    return lock


def evaluate(data, pool, scorer):
    comp_summary, comp = composition_metrics(data, scorer)
    ret_summary, ret = exact_retrieval(pool.images, pool.texts, pool.owner.numpy(), scorer)
    raw = {**comp, **ret, "i2t_cluster_ids": np.asarray(pool.image_ids),
           "t2i_cluster_ids": np.asarray(pool.image_ids)[pool.owner.numpy()],
           "gallery_image_ids": np.asarray(pool.image_ids), "gallery_text_ids": np.asarray(pool.text_ids)}
    for name in ("original", "source_pair"):
        raw[f"{name}_correct"], raw[f"{name}_cluster_ids"] = raw[f"{name}_joint_accuracy"], raw[f"{name}_image_ids"]
    return {"composition": comp_summary, "retrieval": ret_summary}, raw


def reconstruct_gate(result):
    """Rebuild the continuation decision from pinned raw outcome archives."""
    arrays = {}
    for name, entry in result["artifacts"].items():
        with np.load(verify_record(entry), allow_pickle=False) as archive:
            arrays[name] = {key: archive[key] for key in archive.files}
    changes, paired = paired_changes(arrays["frozen"], arrays["trained"])
    exact = exact_changes(paired)
    if changes != result["paired_changes"] or exact != result["exact_paired_changes"]:
        raise ValueError("Official-development reported gains differ from exact outcome counts")
    if set(paired) != set(arrays["paired"]) or any(not np.array_equal(paired[key], arrays["paired"][key]) for key in paired):
        raise ValueError("Official-development saved paired arrays disagree with outcomes")
    _, reconstructed_samples = selected_uncertainty(paired)
    for metric in ("i2t", "t2i", "original", "source_pair"):
        values = arrays["bootstrap"][metric]
        if values.shape != (100000,) or not np.isfinite(values).all():
            raise ValueError("Invalid official-development bootstrap distribution")
        if not np.array_equal(values, reconstructed_samples[metric]):
            raise ValueError("Official-development bootstrap samples differ from deterministic raw-outcome reconstruction")
        lower, upper = np.quantile(values, [.05 / 160, 1 - .05 / 160], method="linear")
        effect = result["effects"][metric]
        if (effect["bootstrap_seed"], effect["replicates"], effect["family_size"]) != (20261007, 100000, 80):
            raise ValueError("Official-development uncertainty recipe changed")
        if effect["ci_lower"] != lower or effect["ci_upper"] != upper or effect["difference"] != float(rational(exact[metric])):
            raise ValueError("Official-development uncertainty summary differs from stored samples")
    checks = {"nonzero_trained_update": result["state"]["update_norm"] > 0,
              "original_joint_improvement": rational(exact["original"]) > 0,
              "source_pair_joint_non_decrease": rational(exact["source_pair"]) >= 0}
    for direction in ("i2t", "t2i"):
        checks[f"{direction}_mean_non_decrease"] = rational(exact[direction]) >= 0
        checks[f"{direction}_strict_adjusted_retention"] = result["effects"][direction]["ci_lower"] > -.01
    if result["state"]["family"] == "labclip":
        identity_changes, identity_paired = paired_changes(arrays["identity"], arrays["trained"])
        identity_exact = exact_changes(identity_paired)
        if (identity_changes != result["attribution"]["learned_vs_identity"]
                or identity_exact != result["attribution"]["exact_learned_vs_identity"]):
            raise ValueError("LABCLIP identity attribution differs from raw outcomes")
        checks["learned_original_joint_improvement_over_identity"] = rational(identity_exact["original"]) > 0
    expected = {"passed": all(checks.values()), "checks": checks}
    if expected != result["gate"]:
        raise ValueError("Stored official-development gate differs from reconstruction")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    lock_args = sub.add_parser("lock")
    lock_args.add_argument("--protocol", required=True)
    lock_args.add_argument("--protocol-sha256", required=True)
    lock_args.add_argument("--selection", required=True)
    lock_args.add_argument("--runs", nargs=2, required=True)
    lock_args.add_argument("--output", required=True)
    evaluate_args = sub.add_parser("evaluate")
    evaluate_args.add_argument("--lock", required=True)
    evaluate_args.add_argument("--lock-sha256", required=True)
    evaluate_args.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    evaluate_args.add_argument("--output", required=True)
    assemble_args = sub.add_parser("assemble")
    assemble_args.add_argument("--lock", required=True)
    assemble_args.add_argument("--lock-sha256", required=True)
    assemble_args.add_argument("--results", nargs=2, required=True)
    assemble_args.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "lock":
        lock = build_lock(args.protocol, args.protocol_sha256, args.selection, args.runs)
        print(json.dumps(write_json(args.output, lock)), flush=True)
        return
    lock = load_lock(args.lock, args.lock_sha256)
    if args.mode == "assemble":
        if lock["seed"] != 17:
            raise ValueError("Individual replication results cannot authorize continuation; use fixed-three-seed aggregation")
        results = {}
        for filename in args.results:
            value = read(filename)
            if value["study"] != "sanw_practical_v9_official_development_result" or value["lock"]["sha256"] != args.lock_sha256:
                raise ValueError("Official-development result lock mismatch")
            encoder = value["encoder"]
            if encoder not in lock["states"] or encoder in results or value["state"] != lock["states"][encoder]:
                raise ValueError("Official-development result state mismatch")
            results[encoder] = {"result": record(filename), "gate": reconstruct_gate(value), "state": value["state"]}
        if set(results) != {"vit_b32", "rn50"}:
            raise ValueError("Both encoder official-development results are required")
        passed_encoders = [name for name in ("vit_b32", "rn50") if results[name]["gate"]["passed"]]
        gate = {"study": "sanw_practical_v9_replication_gate", "family": lock["family"],
                "protocol_sha256": lock["protocol"]["sha256"], "selection_sha256": lock["selection"]["sha256"],
                "lock": record(args.lock), "seed": 17, "passed": len(passed_encoders) == 2,
                "encoders_passed": passed_encoders, "encoders": results,
                "no_new_benchmark_access": True, "status": "official_development_only_not_practical_benchmark_success"}
        print(json.dumps(write_json(args.output, gate)), flush=True)
        return
    torch.set_num_threads(1)
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite official-development evidence")
    protocol = read(lock["protocol"]["path"])
    identity, scorer, identity_scorer, state = verify_full_pilot(lock["states"][args.encoder]["run"], protocol,
        lock["protocol"]["sha256"], lock["selection"]["path"])
    data, pool = load_development(identity, protocol)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "start.json", {"lock": record(args.lock), "state": state, "no_new_benchmark_access": True})
    summaries, raws, artifacts = {}, {}, {}
    for name, model in (("frozen", None), ("trained", scorer)):
        summaries[name], raws[name] = evaluate(data, pool, model)
        artifacts[name] = write_npz(output / f"{name}.npz", raws[name])
    changes, paired = paired_changes(raws["frozen"], raws["trained"])
    exact = exact_changes(paired)
    effects, samples = selected_uncertainty(paired)
    for value in effects.values():
        value["conditioning"] = "one preselected full-fit state fixed; official-development image clusters resampled"
    artifacts["paired"] = write_npz(output / "paired.npz", paired)
    artifacts["bootstrap"] = write_npz(output / "bootstrap.npz", samples)
    checks = {"nonzero_trained_update": state["update_norm"] > 0,
              "original_joint_improvement": rational(exact["original"]) > 0,
              "source_pair_joint_non_decrease": rational(exact["source_pair"]) >= 0}
    for direction in ("i2t", "t2i"):
        checks[f"{direction}_mean_non_decrease"] = rational(exact[direction]) >= 0
        checks[f"{direction}_strict_adjusted_retention"] = effects[direction]["ci_lower"] > -.01
    attribution = None
    if identity_scorer is not None:
        summaries["identity"], identity_raw = evaluate(data, pool, identity_scorer)
        artifacts["identity"] = write_npz(output / "identity.npz", identity_raw)
        identity_changes, identity_paired = paired_changes(identity_raw, raws["trained"])
        identity_exact = exact_changes(identity_paired)
        checks["learned_original_joint_improvement_over_identity"] = rational(identity_exact["original"]) > 0
        parity, parity_raw = paired_changes(raws["frozen"], identity_raw)
        attribution = {"learned_vs_identity": identity_changes, "exact_learned_vs_identity": identity_exact,
                       "identity_vs_frozen": parity, "exact_identity_vs_frozen": exact_changes(parity_raw)}
        artifacts["identity_paired"] = write_npz(output / "identity_paired.npz", identity_paired)
    result = {"study": "sanw_practical_v9_official_development_result", "encoder": args.encoder,
              "lock": record(args.lock), "state": state, "summaries": summaries, "paired_changes": changes,
              "exact_paired_changes": exact, "effects": effects, "attribution": attribution,
              "artifacts": artifacts, "gate": {"passed": all(checks.values()), "checks": checks},
              "evaluation_status": lock["evaluation_status"], "no_new_benchmark_access": True}
    load_lock(args.lock, args.lock_sha256)
    print(json.dumps({"result": write_json(output / "result.json", result), "gate": result["gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
