#!/usr/bin/env python3
"""Verify saved v7 pilot development arrays and the continuation decision."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_development_v7 import select_calibrated_development
from gcr.training import sha256_file


def rec(filename):
    return {"path": str(filename.relative_to(ROOT)), "sha256": sha256_file(filename)}


def main():
    protocol = ROOT / "results/practical_v7/token_source_pair_protocol_v1.json"
    verified, summaries, rows_out = set(), [], []
    checks = 0
    def verify(item):
        key = (item["path"], item["sha256"])
        if key not in verified:
            if sha256_file(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Changed artifact: {item['path']}")
            verified.add(key)
    def same(a, b):
        nonlocal checks
        checks += 1
        if not np.allclose(a, b, atol=1e-15, rtol=0):
            raise ValueError("Saved development aggregate differs from raw arrays")
    for encoder in ("vit_b32", "rn50"):
        base = ROOT / "results/practical_v7/canonical_development_pilot" / encoder
        completion = json.loads((base / "completion.json").read_text())
        if len(completion["snapshots"]) != 1:
            raise ValueError("Pilot scope must contain only seed17")
        snapshot_record = completion["snapshots"][0]; verify(snapshot_record)
        snapshot = json.loads((ROOT / snapshot_record["path"]).read_text())
        if snapshot["seed"] != 17 or len(snapshot["candidates"]) != 24:
            raise ValueError("Pilot epoch/alpha scope differs")
        if select_calibrated_development(snapshot["candidates"]) != snapshot["selection"]:
            raise ValueError("Selection does not reproduce")
        binding = json.loads((ROOT / snapshot["binding"]["path"]).read_text())
        verify(snapshot["binding"]); verify(binding["protocol"])
        for name, sha in binding["source_sha256"].items():
            verify({"path": name, "sha256": sha})
        for item in snapshot["baseline"].values():
            if isinstance(item, dict) and "path" in item:
                verify(item)
        frozen = np.load(ROOT / snapshot["baseline"]["retrieval_predictions"]["path"], allow_pickle=False)
        for row in snapshot["candidates"]:
            for key in ("checkpoint", "composition_predictions", "retrieval_predictions", "bootstrap_predictions", "canonical_delta"):
                verify(row[key])
            comp = np.load(ROOT / row["composition_predictions"]["path"], allow_pickle=False)
            ret = np.load(ROOT / row["retrieval_predictions"]["path"], allow_pickle=False)
            boot = np.load(ROOT / row["bootstrap_predictions"]["path"], allow_pickle=False)
            for archive in (comp, ret, boot):
                for name in archive.files:
                    values = archive[name]
                    if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                        raise ValueError("Nonfinite development raw array")
            same(comp["paired_joint_accuracy"].mean(), row["composition"]["paired_joint_accuracy"])
            same(comp["paired_joint_margin"].mean(), row["composition"]["mean_paired_joint_margin"])
            same(comp["triplet_count"].sum(), row["composition"]["triplet_count"])
            if not np.array_equal(ret["image_correct"], ret["owner"][ret["image_top_indices"][:, 0]] == np.arange(len(ret["image_correct"]))):
                raise ValueError("I2T correctness differs from source ownership")
            if not np.array_equal(ret["text_correct"], ret["text_top_indices"][:, 0] == ret["owner"]):
                raise ValueError("T2I correctness differs from source ownership")
            reasons = []
            gain = row["composition"]["paired_joint_accuracy"] - row["frozen_composition"]["paired_joint_accuracy"]
            if gain <= 0:
                reasons.append("nonpositive_source_pair_joint_gain")
            out = {"encoder": encoder, "epoch": row["epoch"], "alpha": row["alpha"], "joint_gain": gain}
            for direction, name in (("i2t", "image_correct"), ("t2i", "text_correct")):
                effect = row["retention"][direction]
                same((ret[name].astype(int) - frozen[name]).mean(), effect["difference"])
                same(np.quantile(boot[direction], [.05 / 160, 1 - .05 / 160], method="linear"), [effect["ci_lower"], effect["ci_upper"]])
                if effect["difference"] < 0:
                    reasons.append(direction + "_negative_mean")
                if effect["ci_lower"] <= -.01:
                    reasons.append(direction + "_adjusted_lower_not_strictly_above_minus_one_pp")
                out[direction] = {key: effect[key] for key in ("difference", "ci_lower", "ci_upper")}
            out["exclusion_reasons"] = reasons; rows_out.append(out)
            comp.close(); ret.close(); boot.close()
        frozen.close()
        summaries.append({"encoder": encoder, "snapshot": snapshot_record,
                          "completion": rec(base / "completion.json"), "selection": snapshot["selection"]})
    continuation = all(item["selection"]["selected_epoch"] is not None for item in summaries)
    receipt = {"schema": "practical_v7_pilot_development_decision_v1", "protocol": rec(protocol),
               "status": "continue_to_predeclared_replications" if continuation else "stop_after_pilot_development_failure",
               "replication_authorized_by_development_rule": continuation, "pilots": summaries,
               "candidate_count": len(rows_out), "candidates": rows_out,
               "integrity": {"hash_verified_artifacts": len(verified), "numeric_aggregate_checks": checks,
                             "finite_raw_arrays": True, "retrieval_correctness_matches_ownership": True,
                             "confidence_bounds_recomputed_from_saved_samples": True},
               "held_out_evaluation_performed": False,
               "scientific_conclusion": "No deployment claim follows from development; the unchanged practical test gate remains unmet.",
               "audit_source": rec(Path(__file__).resolve())}
    target = ROOT / "results/practical_v7/pilot_development_decision.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt": rec(target), "integrity": receipt["integrity"]}), flush=True)


if __name__ == "__main__":
    main()
