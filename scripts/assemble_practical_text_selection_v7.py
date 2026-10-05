#!/usr/bin/env python3
"""Assemble v7 evaluator manifests only after all six dev selections pass.

The evaluation CLI separately creates its production-format selection lock.
This script creates immutable manifests and a selection assembly receipt.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_development_v7 import ALPHA_GRID, select_calibrated_development
from gcr.training import sha256_file
from calibrate_practical_text_development_v7 import validate_v7_ledger
from rescore_practical_text_development_v6 import (
    candidate_ledger, immutable_json, path, read, record, verify,
)

FAMILY = "bounded_token_source_pair_calibrated_v7"
STUDY = "sanw_practical_source_pair_token_v7"
FACTORY = "gcr.practical_source_pair_v7.SourcePairTrainingExamples"
STATUS = "adaptive_exploratory_after_historical_and_v6_test_reuse"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--canonical-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_record = {"path": args.protocol, "sha256": args.protocol_sha256}
    protocol_path = verify(protocol_record); protocol_record = record(protocol_path)
    protocol = read(protocol_path)
    for name, expected in protocol["source_hashes"].items():
        verify({"path": name, "sha256": expected})
    if protocol["encoders"] != ["vit_b32", "rn50"] or protocol["seeds"] != [17, 29, 43]:
        raise ValueError("Final assembly requires the full predeclared encoder/seed scope")
    canonical_roots, output = [path(value) for value in args.canonical_root], path(args.output)
    manifests, final_states = [], []
    for encoder in protocol["encoders"]:
        # Pilot and later replications can have distinct completion receipts.
        # Each seed is accepted only through one complete canonical snapshot.
        snapshot_paths = sorted(filename for root in canonical_roots for filename in (root / encoder).glob("*/snapshot.json"))
        states, snapshots, seeds, metadata = [], [], set(), None
        if len(snapshot_paths) != 3:
            raise ValueError("Exactly three complete calibration trajectories per encoder are required")
        for snapshot_path in snapshot_paths:
            snapshot = read(snapshot_path)
            if snapshot["encoder"] != encoder or snapshot["seed"] in seeds or snapshot["seed"] not in protocol["seeds"]:
                raise ValueError("Wrong or duplicated seed in final canonical selection")
            seeds.add(snapshot["seed"])
            if not snapshot["trajectory_complete"] or not snapshot["selection_is_finalizable"]:
                raise ValueError("A canonical calibration trajectory is provisional")
            rows = snapshot["candidates"]
            expected = {(epoch, alpha) for epoch in range(1, 5) for alpha in ALPHA_GRID}
            if {(row["epoch"], row["alpha"]) for row in rows} != expected or len(rows) != len(expected):
                raise ValueError("Incomplete epoch-by-alpha candidate grid")
            selection = select_calibrated_development(rows)
            if selection != snapshot["selection"] or selection["selected_epoch"] is None:
                raise ValueError("No reproducible positive-gain feasible development selection")
            binding = read(verify(snapshot["binding"]))
            if binding["protocol"] != protocol_record:
                raise ValueError("Canonical trajectory has a different protocol")
            ledger_path = verify(binding["ledger"])
            ledger = candidate_ledger(ledger_path.parent)
            validate_v7_ledger(ledger, protocol, protocol_record, encoder)
            if ledger["identity"]["config"]["seed"] != snapshot["seed"] or ledger["ledger_sha256"] != binding["ledger_sha256"]:
                raise ValueError("Snapshot seed or ledger identity differs from fitted state")
            for item in binding["inputs"].values():
                verify(item)
            current_metadata = binding["inputs"]["metadata"]
            if metadata is not None and metadata != current_metadata:
                raise ValueError("Feature provenance differs between seeds")
            metadata = current_metadata
            verify(binding["canonical_prefix_receipt"])
            for key in ("composition_predictions", "retrieval_predictions", "checkpoint"):
                verify(snapshot["baseline"][key])
            verified = set()
            for row in rows:
                for key in ("checkpoint", "composition_predictions", "retrieval_predictions", "bootstrap_predictions", "canonical_delta"):
                    item = row[key]
                    signature = (item["path"], item["sha256"])
                    if signature not in verified:
                        verify(item); verified.add(signature)
            chosen = next(row for row in rows if row["epoch"] == selection["selected_epoch"] and row["alpha"] == selection["selected_alpha"])
            state = {"state_id": f"source_pair__{encoder}__seed_{snapshot['seed']}__epoch_{chosen['epoch']:02d}__alpha_{chosen['alpha']:g}",
                     "seed": snapshot["seed"], "epoch": chosen["epoch"], "alpha": chosen["alpha"],
                     "update_norm": chosen["update_norm"], "optimizer_steps": chosen["optimizer_steps"],
                     "checkpoint": chosen["checkpoint"]["path"], "checkpoint_sha256": chosen["checkpoint"]["sha256"],
                     "training_ledger": record(ledger_path), "canonical_development_snapshot": record(snapshot_path),
                     "canonical_development_residual_rms": chosen["residual_rms"],
                     "development_gains": {"paired_joint_accuracy_gain": selection["paired_joint_accuracy_gain"],
                                           "i2t_r1_change": chosen["retention"]["i2t"]["difference"],
                                           "t2i_r1_change": chosen["retention"]["t2i"]["difference"]},
                     "development_retention": chosen["retention"], "canonical_delta": chosen["canonical_delta"]}
            states.append(state); snapshots.append(record(snapshot_path)); final_states.append({"encoder": encoder, **state})
        if seeds != set(protocol["seeds"]):
            raise ValueError("Final manifest is missing required fitted seeds")
        sources = dict(protocol["source_hashes"])
        sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256_file(Path(__file__))
        manifest = {"schema": "sanw_source_pair_token_evaluation_manifest_v7", "encoder": encoder,
                    "family": FAMILY, "study_schema": STUDY, "source_factory": FACTORY,
                    "protocol_sha256": args.protocol_sha256, "protocol": protocol_record,
                    "source_hashes": sources, "training_metadata": metadata,
                    "test_outcomes_used_for_selection": False, "current_test_outcomes_used_for_selection": False,
                    "historical_and_v6_test_exposure": True, "evaluation_status": STATUS,
                    "states": sorted(states, key=lambda state: state["seed"]),
                    "canonical_development_snapshots": snapshots}
        manifest_path = output / f"{encoder}_manifest.json"
        immutable_json(manifest_path, manifest); manifests.append(record(manifest_path))
    immutable_json(output / "selection_assembly.json", {"schema": "sanw_source_pair_token_selection_assembly_v7",
                    "protocol": protocol_record, "family": FAMILY, "manifests": manifests,
                    "selected_states": final_states, "evaluation_status": STATUS,
                    "historical_and_v6_test_exposure": True, "current_test_outcomes_used_for_selection": False,
                    "next_step": "Create evaluator-format lock before any v7 held-out scoring"})
    print(record(output / "selection_assembly.json"), flush=True)


if __name__ == "__main__":
    main()
