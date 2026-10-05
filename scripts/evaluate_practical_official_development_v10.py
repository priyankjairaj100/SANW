#!/usr/bin/env python3
"""Separately locked v10 official development, with the unchanged practical gate.

The expanded training cache is distinct from the original composition validation
cache. State identity uses checkpoint bytes; derived norms are never lock keys.
No benchmark or fresh-confirmation loader exists in this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
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
from evaluate_practical_official_development_v9 import evaluate
from run_practical_streaming_v10 import verify_replication_gate
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer
from gcr.practical_inner_evaluation_v9 import exact_changes, paired_changes, rational
from gcr.practical_exact_bootstrap_v10 import selected_development_uncertainty as selected_uncertainty, lower_exceeds


def require_evaluation_threads():
    if any(os.environ.get(key) != "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("Official development lock and evaluation require all BLAS/OMP threads fixed to one")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def verify_full_pilot(run, protocol, protocol_hash):
    run = root_path(run)
    ledger, completion = read(run / "ledger.json"), read(run / "completion.json")
    identity = ledger["identity"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != ledger["ledger_sha256"] or completion["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Full pilot ledger identity mismatch")
    encoder, config = identity["encoder"], identity["config"]
    seed = config["seed"]
    if (any(value.get("study") != "sanw_practical_v10" or value.get("family") != "joint"
            or value.get("mode") != "full" or value.get("encoder") != encoder
            or value.get("protocol_sha256") != protocol_hash for value in (identity, completion))
            or encoder not in protocol["encoders"] or seed not in (17, 29, 43)
            or completion["config"] != config or config != {**protocol["fit_config"], "seed": seed}
            or digest(verify_record(identity["protocol"])) != protocol_hash):
        raise ValueError("Full pilot recipe, family, or protocol differs")
    if seed != 17:
        gate_record = identity.get("replication_gate")
        if not gate_record:
            raise ValueError("Replication fit lacks a passing seed-17 gate")
        verify_replication_gate(ROOT, verify_record(gate_record), protocol_hash)
    provenance = identity["training_provenance"]
    if (provenance["inputs"] != protocol["training_inputs"][encoder]
            or provenance["original_training_inputs"] != protocol["original_training_inputs"][encoder]
            or provenance["owner_sample"] != protocol["owner_sample"]
            or provenance["confirmation_owner_lock"] != protocol["confirmation_owner_lock"]
            or provenance["training_image_manifest_indices"] != list(range(6000))
            or provenance["original_training_feature_bytes_identical"] is not True
            or provenance["original_normalized_training_features_identical"] is not True
            or provenance["heldout_used"] is not False
            or identity["fit_gallery_image_count"] != 6000 or identity["fit_gallery_text_count"] != 30000
            or identity["official_development_or_benchmarks_used"] is not False):
        raise ValueError("Expanded fit provenance differs from the locked train-only pool")
    for name in ("training_inputs", "original_training_inputs"):
        for item in protocol[name][encoder].values():
            verify_record(item)
    if identity["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("Fit did not bind all protocol sources")
    for source, expected in protocol["source_sha256"].items():
        if digest(ROOT / source) != expected:
            raise ValueError("A bound source changed")
    history = completion["history"]
    if ([row["epoch"] for row in history] != list(range(1, config["epochs"] + 1))
            or any(not np.isfinite(row["training_objective"]) for row in history)):
        raise ValueError("Full fixed training budget or finite objectives missing")
    chosen = min((row for row in history if row["nonzero"]), key=lambda row: (row["training_objective"], row["epoch"]))
    if (completion["selection"] != "minimum_feasible_nonzero_training_objective_then_earliest_epoch"
            or chosen["epoch"] != completion["selected_epoch"]
            or chosen["training_objective"] != completion["selected_training_objective"]):
        raise ValueError("Selected state does not minimize the prescribed training objective")
    certificate = completion["final_certificate"]
    if not all(certificate.get(key) is True for key in ("ranking_checked_canonically", "ranking_preserved", "feasible_with_tolerance")):
        raise ValueError("Selected state lacks its finite-training certificate")
    rows = completion["checkpoint_history"]
    if (len(rows) != len(history)
            or [{key: value for key, value in row.items() if key != "checkpoint"} for row in rows] != history):
        raise ValueError("Checkpoint history differs from the training history")
    checkpoint = run / completion["selected_checkpoint"]["path"]
    epoch_row = rows[chosen["epoch"] - 1]
    epoch_checkpoint = run / epoch_row["checkpoint"]["path"]
    if (digest(checkpoint) != completion["selected_checkpoint"]["sha256"]
            or digest(epoch_checkpoint) != epoch_row["checkpoint"]["sha256"]
            or epoch_row["checkpoint"]["ledger_sha256"] != ledger["ledger_sha256"]):
        raise ValueError("Checkpoint bytes or ledger binding changed")
    model, epoch_model = ConstrainedBilinearScorer.load(checkpoint), ConstrainedBilinearScorer.load(epoch_checkpoint)
    for key in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient"):
        if not np.array_equal(getattr(model, key), getattr(epoch_model, key)):
            raise ValueError("Selected checkpoint differs from its objective-selected epoch")
    nonzero = bool(np.any(model.coefficient != 0))
    squared_norm = float(np.sum(model.coefficient * model.coefficient, dtype=np.float64))
    if not nonzero or not np.isfinite(squared_norm) or squared_norm > config["radius"] ** 2 * (1 + 128 * np.finfo(np.float64).eps):
        raise ValueError("Invalid nonzero or radius-constrained model")
    scorer = CanonicalScorer(model.image_mean, model.text_mean, model.image_basis, model.text_basis, model.coefficient)
    state = {"run": str(run.relative_to(ROOT)), "encoder": encoder, "family": "joint", "seed": seed,
             "epoch": chosen["epoch"], "nonzero": nonzero, "ledger": record(run / "ledger.json"),
             "completion": record(run / "completion.json"), "checkpoint": record(checkpoint)}
    return identity, scorer, state


def build_lock(protocol_path, protocol_hash, runs):
    require_evaluation_threads()
    protocol_path = root_path(protocol_path)
    if digest(protocol_path) != protocol_hash:
        raise ValueError("Protocol hash mismatch")
    protocol = read(protocol_path)
    contract = {**CONTRACT, "selection": "one_preselected_fullfit_state_no_development_search"}
    if protocol.get("study") != "sanw_practical_v10" or protocol["development_contract"] != contract:
        raise ValueError("Official-development contract changed")
    states = {}
    for run in runs:
        _, _, state = verify_full_pilot(run, protocol, protocol_hash)
        if state["encoder"] in states:
            raise ValueError("Duplicate pilot encoder")
        states[state["encoder"]] = state
    if set(states) != {"vit_b32", "rn50"} or len({state["seed"] for state in states.values()}) != 1:
        raise ValueError("Both encoders and one common fixed seed required")
    for encoder, entries in protocol["development_inputs"].items():
        for item in entries.values():
            verify_record(item)
        if {key: entries[key] for key in ("manifest", "features", "metadata")} != protocol["original_training_inputs"][encoder]:
            raise ValueError("Composition validation must retain its original feature archive")
    return {"study": "sanw_practical_v10_official_development_lock", "protocol": record(protocol_path),
            "family": "joint", "seed": next(iter(states.values()))["seed"], "states": states,
            "inputs": protocol["development_inputs"], "contract": contract,
            "extra_sources": {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__),
                              **{name: digest(ROOT / name) for name in (
                                  "scripts/aggregate_practical_official_development_v10.py",
                                  "src/gcr/practical_exact_bootstrap_v10.py",
                                  "scripts/evaluate_practical_official_development_v9.py",
                                  "scripts/aggregate_practical_official_development_v9.py")}},
            "evaluation_status": protocol["evaluation_status"], "no_new_benchmark_access": True}


def load_lock(path, expected):
    if digest(root_path(path)) != expected:
        raise ValueError("Official-development lock hash mismatch")
    lock = read(path)
    rebuilt = build_lock(lock["protocol"]["path"], lock["protocol"]["sha256"], [value["run"] for value in lock["states"].values()])
    if lock != rebuilt:
        raise ValueError("Official-development inputs, states, or sources changed")
    return lock


def gate_checks(exact, effects, nonzero):
    checks = {"nonzero_trained_update": nonzero is True,
              "original_joint_improvement": rational(exact["original"]) > 0,
              "source_pair_joint_non_decrease": rational(exact["source_pair"]) >= 0}
    for direction in ("i2t", "t2i"):
        checks[f"{direction}_mean_non_decrease"] = rational(exact[direction]) >= 0
        checks[f"{direction}_strict_adjusted_retention"] = lower_exceeds(effects[direction], Fraction(-1, 100))
    return {"passed": all(checks.values()), "checks": checks}


def reconstruct_gate(result):
    arrays = {}
    for name in ("frozen", "trained", "paired", "bootstrap"):
        with np.load(verify_record(result["artifacts"][name]), allow_pickle=False) as archive:
            arrays[name] = {key: archive[key] for key in archive.files}
    changes, paired = paired_changes(arrays["frozen"], arrays["trained"])
    exact = exact_changes(paired)
    if changes != result["paired_changes"] or exact != result["exact_paired_changes"]:
        raise ValueError("Reported gains differ from exact outcome counts")
    if set(paired) != set(arrays["paired"]) or any(not np.array_equal(paired[key], arrays["paired"][key]) for key in paired):
        raise ValueError("Saved paired arrays differ from outcome counts")
    effects, samples = selected_uncertainty(paired)
    for metric in ("i2t", "t2i", "original", "source_pair"):
        if not np.array_equal(samples[metric], arrays["bootstrap"][metric]):
            raise ValueError("Saved bootstrap draws differ from deterministic paired reconstruction")
        keys = ("ci_lower", "ci_upper", "difference", "bootstrap_seed", "replicates", "family_size")
        if metric in ("i2t", "t2i"):
            keys += ("exact_ci_lower", "exact_ci_upper")
        for key in keys:
            if effects[metric][key] != result["effects"][metric][key]:
                raise ValueError("Reported adjusted intervals differ from reconstructed bootstrap")
    gate = gate_checks(exact, effects, result["state"]["nonzero"])
    if gate != result["gate"] or gate["passed"] != result["passed"]:
        raise ValueError("Saved gate differs from raw paired evidence")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    lock_args = sub.add_parser("lock")
    lock_args.add_argument("--protocol", required=True); lock_args.add_argument("--protocol-sha256", required=True)
    lock_args.add_argument("--runs", nargs=2, required=True); lock_args.add_argument("--output", required=True)
    evaluate_args = sub.add_parser("evaluate")
    evaluate_args.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    assemble_args = sub.add_parser("assemble")
    assemble_args.add_argument("--results", nargs=2, required=True)
    for command in (evaluate_args, assemble_args):
        command.add_argument("--lock", required=True); command.add_argument("--lock-sha256", required=True)
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    require_evaluation_threads()
    if args.mode == "lock":
        print(json.dumps(write_json(args.output, build_lock(args.protocol, args.protocol_sha256, args.runs))), flush=True)
        return
    lock = load_lock(args.lock, args.lock_sha256)
    if args.mode == "assemble":
        if lock["seed"] != 17:
            raise ValueError("Replication continuation uses the fixed-three-seed mean, not individual gates")
        results, pilot_runs = {}, []
        for filename in args.results:
            result = read(filename); encoder = result["encoder"]
            if (result["study"] != "sanw_practical_v10_official_development" or result["lock"]["sha256"] != args.lock_sha256
                    or encoder not in lock["states"] or encoder in results or result["state"] != lock["states"][encoder]
                    or result["protocol_sha256"] != lock["protocol"]["sha256"] or result["seed"] != 17
                    or result["checkpoint_sha256"] != result["state"]["checkpoint"]["sha256"]):
                raise ValueError("Official-development result state/lock identity differs")
            gate = reconstruct_gate(result)
            results[encoder] = {"result": record(filename), "state": result["state"], "gate": gate}
            # Runner input records deliberately use only path/SHA256; evidence
            # records elsewhere additionally retain descriptive byte counts.
            pilot_runs.append({"encoder": encoder, **{key: {field: value[field] for field in ("path", "sha256")}
                               for key, value in (("result", record(filename)), ("completion", result["state"]["completion"]),
                                                  ("checkpoint", result["state"]["checkpoint"]))}})
        if set(results) != {"vit_b32", "rn50"}:
            raise ValueError("Both pilot encoders required")
        passed = [encoder for encoder in ("vit_b32", "rn50") if results[encoder]["gate"]["passed"]]
        gate = {"study": "sanw_practical_v10_replication_gate", "family": "joint", "seed": 17,
                "protocol_sha256": lock["protocol"]["sha256"], "lock": record(args.lock),
                "passed": len(passed) == 2, "encoders_passed": passed, "encoders": results, "pilot_runs": pilot_runs,
                "no_new_benchmark_access": True, "status": "official_development_only_not_practical_benchmark_success"}
        print(json.dumps(write_json(args.output, gate)), flush=True)
        return
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite official-development evidence")
    protocol = read(lock["protocol"]["path"])
    _, scorer, state = verify_full_pilot(lock["states"][args.encoder]["run"], protocol, lock["protocol"]["sha256"])
    reference = {"encoder": args.encoder, "training_provenance": {"inputs": protocol["original_training_inputs"][args.encoder]}}
    data, pool = load_development(reference, protocol)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "start.json", {"lock": record(args.lock), "state": state, "no_new_benchmark_access": True})
    summaries, raw, artifacts = {}, {}, {}
    for name, model in (("frozen", None), ("trained", scorer)):
        summaries[name], raw[name] = evaluate(data, pool, model)
        artifacts[name] = write_npz(output / f"{name}.npz", raw[name])
    changes, paired = paired_changes(raw["frozen"], raw["trained"])
    exact = exact_changes(paired); effects, samples = selected_uncertainty(paired)
    for value in effects.values():
        value["conditioning"] = "one preselected expanded-fit state fixed; official-development image clusters resampled"
    artifacts["paired"] = write_npz(output / "paired.npz", paired)
    artifacts["bootstrap"] = write_npz(output / "bootstrap.npz", samples)
    gate = gate_checks(exact, effects, state["nonzero"])
    result = {"study": "sanw_practical_v10_official_development", "encoder": args.encoder, "seed": state["seed"],
              "protocol_sha256": lock["protocol"]["sha256"], "checkpoint_sha256": state["checkpoint"]["sha256"],
              "lock": record(args.lock), "state": state, "summaries": summaries, "paired_changes": changes,
              "exact_paired_changes": exact, "effects": effects, "artifacts": artifacts, "gate": gate, "passed": gate["passed"],
              "evaluation_status": lock["evaluation_status"], "no_new_benchmark_access": True}
    load_lock(args.lock, args.lock_sha256)
    print(json.dumps({"result": write_json(output / "result.json", result), "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
