#!/usr/bin/env python3
"""Post-fit optimization diagnosis from the completed, locked v9 training grid.

This reads only protocol, training histories, and fitted states. It never reads
inner-validation predictions, official development, or benchmark results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(protocol_path, fits_root):
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha(protocol_path)
    joint = protocol["joint"]
    expected = {(encoder, radius, weight, joint.get("inner_seed", 17))
                for encoder in protocol["encoders"] for radius in joint["radii"]
                for weight in joint["composition_weights"]}
    rows, found = [], set()
    for path in sorted(fits_root.glob("joint_*/completion.json")):
        result = json.loads(path.read_text())
        if result.get("protocol_sha256") != protocol_sha:
            continue
        if result.get("study") != "sanw_practical_v9" or result.get("family") != "joint" or result.get("mode") != "inner":
            raise ValueError("Unexpected training receipt schema")
        config = result["config"]
        identity = (result["encoder"], config["radius"], config["composition_weight"], config["seed"])
        if identity not in expected or identity in found:
            raise ValueError("Unexpected or repeated locked-grid cell")
        found.add(identity)
        history = result["history"]
        selected = next(row for row in history if row["epoch"] == result["selected_epoch"])
        certificate = selected["certificate"]
        selected_path = path.parent / result["selected_checkpoint"]["path"]
        if sha(selected_path) != result["selected_checkpoint"]["sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        epoch_path = path.parent / "checkpoints" / f"epoch_{result['selected_epoch']:03d}.npz"
        with np.load(selected_path, allow_pickle=False) as a, np.load(epoch_path, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
                raise ValueError("Selected state differs from its recorded epoch")
            actual_norm = float(np.linalg.norm(a["coefficient"]))
        reduction = result["initial_training_objective"] - result["selected_training_objective"]
        upper_gap = selected["ball_relaxed_convex_suboptimality_upper_bound"]
        last = history[-8:]
        rows.append({"encoder": result["encoder"], "radius": config["radius"], "composition_weight": config["composition_weight"],
                     "seed": config["seed"], "selected_epoch": result["selected_epoch"], "epochs": len(history),
                     "optimizer_steps": result["optimizer_steps"], "elapsed_seconds": result["elapsed_seconds"],
                     "initial_objective": result["initial_training_objective"], "selected_objective": result["selected_training_objective"],
                     "objective_reduction": reduction, "ball_relaxed_suboptimality_upper_bound": upper_gap,
                     "upper_gap_divided_by_achieved_reduction": upper_gap / reduction if reduction > 0 else None,
                     "last_eight_objective_range": max(x["training_objective"] for x in last) - min(x["training_objective"] for x in last),
                     "last_epoch_minus_selected_objective": history[-1]["training_objective"] - result["selected_training_objective"],
                     "coefficient_norm": actual_norm, "radius_utilization": actual_norm / config["radius"],
                     "minimum_radial_restoration_factor": min(x["certificate"]["radial_restoration_factor"] for x in history),
                     "selected_joint_gain_pp": 100 * (selected["composition"]["image_mean_joint_accuracy"] - result["baseline_composition"]["image_mean_joint_accuracy"]),
                     "selected_i2t_gain_pp": 100 * (certificate["current_i2t_correct"] - certificate["protected_i2t_queries"]) / certificate["source_gallery_images"],
                     "selected_t2i_gain_pp": 100 * (certificate["current_t2i_correct"] - certificate["protected_t2i_queries"]) / certificate["source_gallery_texts"],
                     "lost_originally_correct_i2t": certificate["lost_frozen_correct_i2t"],
                     "lost_originally_correct_t2i": certificate["lost_frozen_correct_t2i"],
                     "checked_training_constraints": certificate["checked_constraints"],
                     "canonical_rankings_checked": certificate["ranking_checked_canonically"],
                     "active_constraints": certificate["active_constraints"], "fixed_composition_multiplier": result["fixed_composition_multiplier"],
                     "completion": {"path": str(path), "sha256": sha(path)},
                     "selected_checkpoint": {"path": str(selected_path), "sha256": sha(selected_path)}})
    if found != expected:
        raise ValueError(f"Grid incomplete: {len(found)} completed of {len(expected)} locked cells")
    return {"study": "sanw_practical_v9_training_diagnosis", "scope": "training_only_postfit_no_updates_or_selection",
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha}, "fits": len(rows),
            "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
            "training_constraints_checked_at_selected_states": sum(row["checked_training_constraints"] for row in rows),
            "all_selected_training_rankings_preserved": all(row["lost_originally_correct_i2t"] == 0 and row["lost_originally_correct_t2i"] == 0 and row["canonical_rankings_checked"] for row in rows),
            "suboptimality_bound": "Convex first-order inequality and Frobenius-ball relaxation. A small bound certifies proximity in this training objective. A large upper bound does not prove suboptimality.",
            "limitations": "No validation or benchmark success follows from training objective values, gains, or finite-training certificates.",
            "diagnostic_source_sha256": sha(Path(__file__)), "cells": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fits-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a previous diagnosis")
    result = diagnose(args.protocol.resolve(), args.fits_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("fits", "optimizer_steps", "training_constraints_checked_at_selected_states", "all_selected_training_rankings_preserved")}))


if __name__ == "__main__":
    main()
