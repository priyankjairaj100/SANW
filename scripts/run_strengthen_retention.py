#!/usr/bin/env python3
"""Run the frozen retention v3 grid using training/development caches only."""
import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gcr.review_training import SourceRetrievalPool
from gcr.strengthen_retention import (POLICIES, PROTOCOL_SHA256, build_ledger, ensure_ledger, fit_candidate,
    fit_matched_controls, original_config, select_and_export)
from gcr.training import FeatureDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("assignments", "plan", "fit", "select", "matched-controls", "all"))
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), default="vit_b32")
    parser.add_argument("--protocol", default="results/strengthen_retention/protocol_v3.json")
    parser.add_argument("--output")
    parser.add_argument("--train-manifest")
    parser.add_argument("--train-features")
    parser.add_argument("--train-metadata")
    parser.add_argument("--dev-manifest", default="data/review_followup/e_vil_dev900/manifest.json")
    parser.add_argument("--dev-features")
    parser.add_argument("--dev-metadata")
    parser.add_argument("--assignment-directory")
    parser.add_argument("--assignment-parity-receipt")
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(17, 29, 43))
    parser.add_argument("--only-learning-rate", type=float)
    args = parser.parse_args()
    original, config = original_config(REPOSITORY, 512 if args.encoder == "vit_b32" else 1024)
    original_inputs = original["identity"]["inputs"]
    if args.encoder == "rn50" and any(value is None for value in (args.train_features, args.train_metadata, args.dev_features, args.dev_metadata, args.assignment_directory)):
        parser.error("RN50 requires explicit train/development caches and RN50-specific assignment directory")
    train_manifest = REPOSITORY / (args.train_manifest or original_inputs["manifest"]["path"])
    train_features = REPOSITORY / (args.train_features or original_inputs["features"]["path"])
    train_metadata = REPOSITORY / (args.train_metadata or original_inputs["metadata"]["path"])
    data = FeatureDataset(REPOSITORY, train_manifest, train_features, train_metadata, config.feature_dim)
    output = REPOSITORY / (args.output or f"results/strengthen_retention/{args.encoder}")
    if args.phase == "assignments":
        if args.encoder != "vit_b32":
            parser.error("RN50 assignments must come from its encoder replication's frozen controls")
        from gcr.strengthen_replication import audit_recovered_control_parity
        receipt = audit_recovered_control_parity(REPOSITORY, output, data, PROTOCOL_SHA256)
        print(json.dumps({"phase": "assignments", "passed": receipt["passed"],
                          "directory": receipt["regenerated_assignment_directory"]}), flush=True)
        return
    pool = SourceRetrievalPool(REPOSITORY, REPOSITORY / args.dev_manifest,
        REPOSITORY / (args.dev_features or "results/review_followup/features/e_vil_dev900/features.npz"),
        REPOSITORY / (args.dev_metadata or "results/review_followup/features/e_vil_dev900/metadata.json"), data, config.feature_dim)
    default_assignments = output / "recovered_assignment_controls"
    if not default_assignments.exists():
        default_assignments = REPOSITORY / "results/review_followup/assignments"
    ledger, assignments = build_ledger(REPOSITORY, output, args.encoder, config, data, pool,
        REPOSITORY / args.protocol, REPOSITORY / args.assignment_directory if args.assignment_directory else default_assignments,
        REPOSITORY / args.assignment_parity_receipt if args.assignment_parity_receipt else None)
    rates = (args.only_learning_rate,) if args.only_learning_rate is not None else config.learning_rates
    if any(rate not in config.learning_rates for rate in rates):
        parser.error("Learning rate is outside frozen protocol")
    policies, seeds = args.policies or POLICIES, args.seeds or config.seeds
    if args.phase == "all" and (args.policies or args.seeds or args.only_learning_rate is not None):
        parser.error("all phase must execute the complete frozen grid")
    if args.phase == "plan":
        print(json.dumps({"phase": "plan", "ledger": ledger, "full_candidates": 81,
                          "requested_candidates": len(policies) * len(seeds) * len(rates)}, indent=2), flush=True)
        return
    ensure_ledger(output, ledger)
    if args.phase in ("fit", "all"):
        for policy in policies:
            for rate in rates:
                for seed in seeds:
                    result = fit_candidate(REPOSITORY, output, config, data, pool, ledger, policy, rate, seed)
                    print(json.dumps({"phase": "fit", **{k: v for k, v in result.items() if k != "artifact_sha256"}}), flush=True)
    if args.phase in ("select", "all"):
        manifest = select_and_export(REPOSITORY, output, config, ledger)
        print(json.dumps({"phase": "select", "state_count": len(manifest["states"]), "selection_count": len(manifest["selections"])}), flush=True)
    if args.phase in ("matched-controls", "all"):
        manifest = fit_matched_controls(REPOSITORY, output, config, data, pool, ledger, assignments)
        print(json.dumps({"phase": "matched-controls", "state_count": len(manifest["states"]),
                          "selection_count": len(manifest["selections"]), "complete": manifest["matched_controls_complete"]}), flush=True)


if __name__ == "__main__":
    main()
