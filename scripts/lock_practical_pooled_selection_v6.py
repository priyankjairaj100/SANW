#!/usr/bin/env python3
"""Validate canonical pooled-family development evidence and lock six states.

This command never loads held-out benchmark features or performs held-out scoring.
All selected states require complete canonical development trajectories.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_practical_v6 import create_lock, immutable
from gcr.practical_training import select_development_epoch
from gcr.training import sha256_file
from rescore_practical_development_v6 import (
    content_hash, immutable_json, load_checkpoint, read, relative, root_path,
    verify_candidate, verify_record,
)

SEEDS = (17, 29, 43)
ENCODERS = ("vit_b32", "rn50")


def record(filename):
    filename = root_path(filename)
    return {"path": relative(filename), "sha256": sha256_file(filename)}


def arrays(artifact):
    with np.load(verify_record(artifact), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def verify_predictions(row):
    composition = arrays(row["composition"]["predictions"])
    retrieval = arrays(row["retrieval"]["predictions"])
    if float(composition["paired_joint_accuracy"].mean()) != row["composition"]["paired_joint_accuracy"]:
        raise ValueError("Composition metric differs from raw predictions")
    if float(composition["paired_joint_margin"].mean()) != row["composition"]["mean_paired_joint_margin"]:
        raise ValueError("Composition tie-break margin differs from raw predictions")
    expected_i = retrieval["owner"][retrieval["i2t_top_indices"]] == np.arange(len(retrieval["image_ids"]))
    expected_t = retrieval["t2i_top_indices"] == retrieval["owner"]
    for direction, expected, prefix in (("i2t", expected_i, "image"), ("t2i", expected_t, "text")):
        if not np.array_equal(expected, retrieval[f"{direction}_correct"]):
            raise ValueError("Retrieval ownership correctness differs")
        if int(expected.sum()) != row["retrieval"][f"{prefix}_correct_count"]:
            raise ValueError("Retrieval count differs from raw predictions")
        if float(expected.mean()) != row["retrieval"][f"{direction}_r1"]:
            raise ValueError("Retrieval accuracy differs from raw predictions")
    return composition, retrieval


def residual_diagnostics(selected_composition, selected_retrieval, baseline_composition, epsilon):
    for key in ("raw_image_index", "raw_text_index", "raw_relation"):
        if not np.array_equal(selected_composition[key], baseline_composition[key]):
            raise ValueError("Canonical paired identities differ")
    delta = selected_composition["raw_score"] - baseline_composition["raw_score"]
    centered = delta.copy()
    for image_id in np.unique(selected_composition["raw_image_index"]):
        mask = selected_composition["raw_image_index"] == image_id
        centered[mask] -= centered[mask].mean()
    ret = selected_retrieval["union_residuals"]
    for values in (delta, ret):
        if not np.isfinite(values).all() or np.max(np.abs(values)) > epsilon + 1e-12:
            raise ValueError("Saved canonical residual exceeds model bound")
    metrics = {
        "composition_pairs": len(delta), "composition_rms": float(np.sqrt(np.mean(delta**2))),
        "composition_within_image_centered_rms": float(np.sqrt(np.mean(centered**2))),
        "composition_max_absolute": float(np.max(np.abs(delta))),
        "retrieval_union_pairs": len(ret), "retrieval_rms": float(np.sqrt(np.mean(ret**2))),
        "retrieval_std": float(np.std(ret)), "retrieval_max_absolute": float(np.max(np.abs(ret))),
        "nonconstant_numerical_tolerance": 1e-12,
        "population": "development pairs only; no held-out data used",
    }
    if metrics["composition_within_image_centered_rms"] <= 1e-12 or metrics["retrieval_std"] <= 1e-12:
        raise ValueError("Selected correction is effectively zero or constant on development pairs")
    return metrics


def select_state(encoder, seed, args, plan, plan_record, amendment_record):
    name = f"eps_{plan['epsilon']:g}_ret_{plan['retention_weight']:g}_seed_{seed}"
    candidate = root_path(args.candidates_root) / encoder / name
    canonical = root_path(args.canonical_root) / encoder / name
    ledger, original_history = verify_candidate(candidate)
    identity = ledger["identity"]
    config = identity["config"]
    if config["seed"] != seed or config["encoder"] != encoder or config["epsilon"] != plan["epsilon"] or config["retention_weight"] != plan["retention_weight"]:
        raise ValueError("Candidate differs from fixed common-configuration plan")
    completion_record = record(candidate / "completion.json")
    completion = read(candidate / "completion.json")
    if completion["status"] != "completed_development_only" or completion["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Candidate fitting did not complete")
    if completion["history_sha256"] != sha256_file(candidate / "history.json") or completion["epochs"] != plan["epochs"]:
        raise ValueError("Completed fitting history differs")
    full = []
    for filename in sorted(canonical.glob("selection_*.json")):
        snapshot = read(filename)
        if snapshot["trajectory_complete"] and snapshot["selection_is_finalizable"]:
            full.append((filename, snapshot))
    if not full:
        raise ValueError(f"No complete canonical trajectory: {canonical}")
    if len({content_hash(value["epochs"]) for _, value in full}) != 1:
        raise ValueError("Conflicting complete canonical trajectories")
    snapshot_path, snapshot = full[0]
    rows = snapshot["epochs"]
    if [row["epoch"] for row in rows] != list(range(plan["epochs"] + 1)):
        raise ValueError("Canonical trajectory misses an epoch")
    selected = select_development_epoch(rows)
    if selected != snapshot["selection"] or selected.get("selected_epoch") is None:
        raise ValueError("Canonical selection is infeasible or cannot be reproduced")
    if snapshot["current_test_outcomes_used_for_selection"] is not False:
        raise ValueError("Selection cannot use current test outcomes")
    binding = read(verify_record(snapshot["binding"]))
    if binding["amendment"] != amendment_record or binding["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Canonical inference amendment or ledger differs")
    evidence = {relative(snapshot_path): sha256_file(snapshot_path), completion_record["path"]: completion_record["sha256"]}
    for key in ("ledger", "amendment", "protocol", "shortlist"):
        filename = verify_record(binding[key])
        evidence[relative(filename)] = sha256_file(filename)
    for filename, expected in binding["source_hashes"].items():
        verify_record({"path": filename, "sha256": expected})
    for value in binding["inputs"].values():
        filename = verify_record(value)
        evidence[relative(filename)] = sha256_file(filename)
    shortlist = read(binding["shortlist"]["path"])
    filename = verify_record(shortlist["archive"])
    evidence[relative(filename)] = sha256_file(filename)
    raw = {}
    for row in rows:
        verify_record(row["checkpoint"])
        if row["binding"] != snapshot["binding"]:
            raise ValueError("Epoch is bound to a different canonical inference record")
        raw[row["epoch"]] = verify_predictions(row)
        for endpoint in ("composition", "retrieval"):
            artifact = row[endpoint]["predictions"]
            evidence[artifact["path"]] = artifact["sha256"]
    epoch = selected["selected_epoch"]
    row, baseline = rows[epoch], rows[0]
    if selected["i2t_r1_change"] < 0 or selected["t2i_r1_change"] < 0:
        raise ValueError("Selected development retrieval declined")
    original_rows = {value["epoch"]: value for value in original_history["epochs"]}
    initial_model, _, _ = load_checkpoint(candidate, original_rows[0], ledger, None)
    initial = {name: value.detach().clone() for name, value in initial_model.named_parameters()}
    model, payload, checkpoint = load_checkpoint(candidate, original_rows[epoch], ledger, initial)
    if checkpoint != row["checkpoint"] or payload["update_norm"] != row["update_norm"]:
        raise ValueError("Canonical selected model differs from fitted checkpoint")
    diagnostics = residual_diagnostics(raw[epoch][0], raw[epoch][1], raw[0][0], model.epsilon)
    state = {
        "state_id": f"{encoder}_bounded_pair_seed_{seed}_epoch_{epoch:02d}", "seed": seed, "epoch": epoch,
        "checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"],
        "update_norm": payload["update_norm"], "optimizer_steps": payload["optimizer_steps"],
        "model_config": payload["model_config"], "development_selection": selected,
        "canonical_development_snapshot": record(snapshot_path), "canonical_inference_binding": snapshot["binding"],
        "training_completion": completion_record, "training_ledger": record(candidate / "ledger.json"),
        "nonzero_diagnostics": diagnostics,
    }
    return state, identity, binding, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-root", default="results/practical_v6")
    parser.add_argument("--canonical-root", default="results/practical_v6/canonical_development")
    parser.add_argument("--replication-plan", default="results/practical_v6/common_configuration_replication_plan.json")
    parser.add_argument("--replication-plan-sha256", required=True)
    parser.add_argument("--amendment", default="results/practical_v6/inference_amendment_v1.json")
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--output", default="results/practical_v6/pooled_selection")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan_record = {"path": relative(args.replication_plan), "sha256": args.replication_plan_sha256}
    amendment_record = {"path": relative(args.amendment), "sha256": args.amendment_sha256}
    plan, amendment = read(verify_record(plan_record)), read(verify_record(amendment_record))
    for filename, expected in plan["source_sha256"].items():
        verify_record({"path": filename, "sha256": expected})
    verify_record(plan["selection_evidence"])
    output = root_path(args.output)
    manifests = []
    for encoder in ENCODERS:
        states, identities, bindings, evidence = [], [], [], {}
        for seed in SEEDS:
            state, identity, binding, own_evidence = select_state(encoder, seed, args, plan, plan_record, amendment_record)
            states.append(state)
            identities.append(identity)
            bindings.append(binding)
            evidence.update(own_evidence)
        if len({content_hash(i["inputs"]) for i in identities}) != 1:
            raise ValueError("Seed feature provenance differs")
        sources = {}
        for identity, binding in zip(identities, bindings):
            sources.update(identity["source_sha256"])
            sources.update(binding["source_hashes"])
        sources[relative(__file__)] = sha256_file(Path(__file__))
        sources["scripts/audit_practical_v6.py"] = sha256_file(ROOT / "scripts/audit_practical_v6.py")
        manifest = {
            "schema_version": 1, "encoder": encoder, "family": plan["family"],
            "protocol": amendment["base_protocol"], "protocol_sha256": amendment["base_protocol_sha256"],
            "inference_amendment": amendment_record, "replication_plan": plan_record,
            "configuration_selection_evidence": plan["selection_evidence"],
            "evaluation_status": "exploratory_after_historical_test_reuse",
            "test_outcomes_used_for_selection": False, "current_test_outcomes_used_for_selection": False,
            "selection_rule": identities[0]["selection_rule"], "all_canonical_trajectories_complete": True,
            "states": states, "source_hashes": sources, "development_evidence_hashes": evidence,
            "training_metadata": identities[0]["inputs"]["metadata"], "feature_inputs": identities[0]["inputs"],
            "practical_gate": read(amendment["base_protocol"])["practical_gate"],
            "held_out_scoring_performed_by_builder": False,
        }
        manifest_path = output / f"{encoder}_manifest.json"
        immutable_json(manifest_path, manifest)
        manifests.append(relative(manifest_path))
    lock = create_lock(amendment["base_protocol"], amendment["base_protocol_sha256"], manifests)
    lock_path = output / "selection_lock.json"
    immutable(lock_path, lock)
    summary = {"status": "six_nonzero_canonical_development_states_locked_no_heldout_scoring",
               "selection_lock": record(lock_path), "manifests": [record(filename) for filename in manifests]}
    immutable_json(output / "completion.json", summary)
    import json
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
