#!/usr/bin/env python3
"""Fixed three-seed development aggregation under the unchanged v8 protocol.

Only existing development prediction archives are read. Both seed-17 pilots
must pass their original gate. Individual replication gates do not have to
pass: the replication gate is evaluated on the fixed selected-seed mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_practical_constrained_development_v8 import (
    CONTRACT, SOURCES, digest, read, record, root_path, verified_run,
    verify_record, write_json, write_npz,
)
from gcr.practical_constrained_evaluation_v8 import development_gate, paired_cluster_bootstrap


SEEDS = (17, 29, 43)
ENCODERS = ("vit_b32", "rn50")


def arrays_from_record(value):
    with np.load(verify_record(value), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_result(filename, protocol_hash, protocol):
    result = read(filename)
    if result["schema"] != "sanw_v8_development_result_v1":
        raise ValueError("Wrong development result schema")
    start = read(verify_record(result["start_receipt"]))
    if start["schema"] != "sanw_v8_development_start_v1" or start["contract"] != CONTRACT:
        raise ValueError("Unrecognized development start receipt")
    if start["protocol"]["sha256"] != protocol_hash:
        raise ValueError("Result belongs to a different protocol")
    if start["no_heldout_benchmark_access"] is not True or start["alpha_search"] or start["checkpoint_search"]:
        raise ValueError("Result does not obey the fixed-state development restriction")
    completion_path = verify_record(start["run"])
    verify_record(start["ledger"])
    verify_record(start["checkpoint"])
    identity, completion, _, model, norm = verified_run(completion_path.parent, protocol_hash)
    if (result["encoder"], result["seed"], result["selected_epoch"]) != (
        identity["encoder"], identity["config"]["seed"], completion["selected_epoch"]
    ):
        raise ValueError("Result does not identify the training-selected state")
    if start["encoder"] != result["encoder"] or start["seed"] != result["seed"]:
        raise ValueError("Result and start receipt identify different states")
    if start["inputs"] != protocol["development_inputs"][identity["encoder"]]:
        raise ValueError("Development input identity differs from the protocol")
    for value in start["inputs"].values():
        verify_record(value)
    for source in SOURCES:
        if start["source_sha256"][source] != digest(ROOT / source):
            raise ValueError("Single-state evaluator source changed")
    raw = {name: arrays_from_record(result["artifacts"][name]) for name in (
        "frozen_composition", "trained_composition", "frozen_retrieval", "trained_retrieval"
    )}
    for state in ("frozen", "trained"):
        comp, retrieval = raw[f"{state}_composition"], raw[f"{state}_retrieval"]
        for kind in ("original", "source_pair"):
            if float(comp[f"{kind}_joint_accuracy"].mean()) != result["summaries"][state]["composition"][kind]["joint_accuracy"]:
                raise ValueError("Composition summary differs from the raw image values")
        for direction in ("i2t", "t2i"):
            if float(retrieval[f"{direction}_correct"].mean()) != result["summaries"][state]["retrieval"][f"{direction}_r1"]:
                raise ValueError("Retrieval summary differs from raw query correctness")
    return {"result": result, "raw": raw, "norm": norm, "identity": record(filename),
            "checkpoint": start["checkpoint"], "start_receipt": result["start_receipt"]}


def fixed_seed_arrays(rows):
    """Validate common units and retain all fixed-seed per-item predictions."""
    if tuple(row["result"]["seed"] for row in rows) != SEEDS:
        raise ValueError("Exactly the fixed ordered seeds 17, 29, 43 are required")
    reference = rows[0]["raw"]
    for row in rows:
        raw = row["raw"]
        for kind in ("composition", "retrieval"):
            name = f"frozen_{kind}"
            if set(raw[name]) != set(reference[name]) or any(
                not np.array_equal(raw[name][key], reference[name][key]) for key in reference[name]
            ):
                raise ValueError("Frozen predictions differ between selected seeds")
        for kind in ("original", "source_pair"):
            for key in (f"{kind}_image_ids", f"{kind}_triplet_counts"):
                if not np.array_equal(raw["trained_composition"][key], reference["frozen_composition"][key]):
                    raise ValueError("Composition image units differ between states")
        for key in ("image_ids", "text_ids", "owner"):
            if not np.array_equal(raw["trained_retrieval"][key], reference["frozen_retrieval"][key]):
                raise ValueError("Retrieval ownership units differ between states")
    result = {"seeds": np.asarray(SEEDS, dtype=np.int64)}
    ret = reference["frozen_retrieval"]
    result.update({key: ret[key] for key in ("image_ids", "text_ids", "owner")})
    for endpoint in ("i2t", "t2i"):
        result[f"{endpoint}_frozen"] = ret[f"{endpoint}_correct"].astype(np.float64)
        values = np.stack([row["raw"]["trained_retrieval"][f"{endpoint}_correct"].astype(np.float64) for row in rows])
        if values.ndim != 2 or not np.isin(values, [0., 1.]).all():
            raise ValueError("Retrieval correctness must be Boolean for each seed/query")
        result[f"{endpoint}_trained_by_seed"] = values
        result[f"{endpoint}_trained_seed_mean"] = values.mean(axis=0)
        result[f"{endpoint}_paired_seed_mean_difference"] = values.mean(axis=0) - result[f"{endpoint}_frozen"]
    for endpoint in ("original", "source_pair"):
        comp = reference["frozen_composition"]
        result[f"{endpoint}_image_ids"] = comp[f"{endpoint}_image_ids"]
        result[f"{endpoint}_triplet_counts"] = comp[f"{endpoint}_triplet_counts"]
        result[f"{endpoint}_frozen"] = comp[f"{endpoint}_joint_accuracy"].astype(np.float64)
        values = np.stack([row["raw"]["trained_composition"][f"{endpoint}_joint_accuracy"] for row in rows])
        if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Composition per-image accuracy is invalid")
        result[f"{endpoint}_trained_by_seed"] = values
        result[f"{endpoint}_trained_seed_mean"] = values.mean(axis=0)
        result[f"{endpoint}_paired_seed_mean_difference"] = values.mean(axis=0) - result[f"{endpoint}_frozen"]
    return result


def aggregate_arrays(arrays, norms):
    effects, samples = {}, {}
    clusters = {"i2t": arrays["image_ids"], "t2i": arrays["image_ids"][arrays["owner"]],
                "original": arrays["original_image_ids"], "source_pair": arrays["source_pair_image_ids"]}
    for endpoint in clusters:
        effects[endpoint], samples[endpoint] = paired_cluster_bootstrap(
            arrays[f"{endpoint}_paired_seed_mean_difference"], clusters[endpoint]
        )
        effects[endpoint]["conditioning"] = "fixed selected seeds 17,29,43 averaged within each query; only image clusters resampled"
    frozen = {name: {"joint_accuracy": float(arrays[f"{name}_frozen"].mean())} for name in ("original", "source_pair")}
    trained = {name: {"joint_accuracy": float(arrays[f"{name}_trained_seed_mean"].mean())} for name in ("original", "source_pair")}
    if len(norms) != 3 or any(not np.isfinite(norm) or norm <= 0 for norm in norms):
        raise ValueError("All three selected states must have nonzero finite updates")
    gate = development_gate(frozen, trained, effects, update_norm=min(norms))
    gate.update({"individual_replication_gate_required": False, "fixed_selected_seeds": list(SEEDS),
                 "seed_resampling": False, "score_ensemble": False})
    return {"effects": effects, "gate": gate,
            "composition": {"frozen": frozen, "fixed_seed_mean": trained}}, samples


def pilot_gate(row):
    """Recompute the seed-17 gate from raw predictions, not a claimed flag."""
    raw = row["raw"]
    frozen, trained = raw["frozen_retrieval"], raw["trained_retrieval"]
    effects = {}
    for direction, clusters in (("i2t", frozen["image_ids"]),
                                ("t2i", frozen["image_ids"][frozen["owner"]])):
        delta = trained[f"{direction}_correct"].astype(np.float64) - frozen[f"{direction}_correct"].astype(np.float64)
        effects[direction], _ = paired_cluster_bootstrap(delta, clusters)
    comp = {state: {kind: {"joint_accuracy": float(raw[f"{state}_composition"][f"{kind}_joint_accuracy"].mean())}
                   for kind in ("original", "source_pair")} for state in ("frozen", "trained")}
    gate = development_gate(comp["frozen"], comp["trained"], effects, update_norm=row["norm"])
    if gate != row["result"]["gate"]:
        raise ValueError("Pilot's stored gate differs from the independently reconstructed raw-array gate")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--results", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol_path = root_path(args.protocol)
    if digest(protocol_path) != args.protocol_sha256:
        raise ValueError("Protocol hash mismatch")
    protocol = read(protocol_path)
    if protocol["development_contract"] != CONTRACT or protocol["seeds"] != list(SEEDS) or protocol["encoders"] != list(ENCODERS):
        raise ValueError("Protocol population or development gate differs")
    if protocol["replication_rule"]["gate"] != "same_development_contract_on_fixed_three_seed_mean_per_encoder":
        raise ValueError("Protocol does not declare this fixed-seed aggregation")
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an aggregation")
    rows = [load_result(filename, args.protocol_sha256, protocol) for filename in args.results]
    identities = [(row["result"]["encoder"], row["result"]["seed"]) for row in rows]
    if set(identities) != {(encoder, seed) for encoder in ENCODERS for seed in SEEDS} or len(set(identities)) != len(identities):
        raise ValueError("Require all six unique encoder/seed states")
    pilots = {row["result"]["encoder"]: pilot_gate(row) for row in rows if row["result"]["seed"] == 17}
    if set(pilots) != set(ENCODERS) or not all(value["passed"] for value in pilots.values()):
        raise ValueError("Both seed-17 pilots must pass before replication aggregation")
    output.mkdir(parents=True, exist_ok=True)
    start = {"schema": "sanw_v8_fixed_seed_development_start_v1", "protocol": record(protocol_path),
             "results": [row["identity"] for row in rows], "checkpoints": [row["checkpoint"] for row in rows],
             "start_receipts": [row["start_receipt"] for row in rows], "development_inputs": protocol["development_inputs"],
             "pilot_gates_reconstructed": pilots, "seeds": list(SEEDS), "encoders": list(ENCODERS),
             "source_sha256": {"scripts/aggregate_practical_constrained_development_v8.py": digest(__file__),
                               **{source: digest(ROOT / source) for source in SOURCES}},
             "no_heldout_access": True, "no_checkpoint_or_alpha_selection": True}
    start_record = write_json(output / "aggregation_start.json", start)
    result, artifacts = {}, {}
    for encoder in ENCODERS:
        selected = sorted([row for row in rows if row["result"]["encoder"] == encoder], key=lambda row: row["result"]["seed"])
        arrays = fixed_seed_arrays(selected)
        result[encoder], samples = aggregate_arrays(arrays, [row["norm"] for row in selected])
        artifacts[f"{encoder}_fixed_seed_predictions"] = write_npz(output / f"{encoder}_fixed_seed_predictions.npz", arrays)
        artifacts[f"{encoder}_bootstrap_samples"] = write_npz(output / f"{encoder}_bootstrap_samples.npz", samples)
    for row in rows:
        verify_record(row["identity"])
        verify_record(row["checkpoint"])
    final = {"schema": "sanw_v8_fixed_seed_development_result_v1", "start_receipt": start_record,
             "encoders": result, "artifacts": artifacts,
             "both_encoders_pass": all(value["gate"]["passed"] for value in result.values()),
             "interpretation": "development only; fixed seed mean performance, not an ensemble; practical test goal remains unevaluated",
             "elapsed_seconds": time.monotonic() - started}
    write_json(output / "result.json", final)
    print(json.dumps({"both_encoders_pass": final["both_encoders_pass"], "encoders": result,
                      "result": record(output / "result.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
