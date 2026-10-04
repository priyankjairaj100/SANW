#!/usr/bin/env python3
"""Run the separately frozen exploratory A+D grid, using development data only."""
import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gcr.allocation_distillation import (
    POLICIES, build_ledger, ensure_ledger, fit_candidate, fit_matched_controls, generate_assignments,
    original_config, protocol_template, select_and_export, verify_protocol,
)
from gcr.review_training import SourceRetrievalPool
from gcr.training import FeatureDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("protocol-template", "assignments", "plan", "fit", "select", "matched-controls", "all"))
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), default="vit_b32")
    parser.add_argument("--protocol", default="results/allocation_distillation/protocol_v1.json")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--output")
    parser.add_argument("--train-manifest")
    parser.add_argument("--train-features")
    parser.add_argument("--train-metadata")
    parser.add_argument("--dev-manifest", default="data/review_followup/e_vil_dev900/manifest.json")
    parser.add_argument("--dev-features")
    parser.add_argument("--dev-metadata")
    parser.add_argument("--assignment-directory")
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(17, 29, 43))
    parser.add_argument("--only-learning-rate", type=float)
    args = parser.parse_args()
    if args.phase == "protocol-template":
        print(json.dumps(protocol_template(), indent=2))
        return
    if args.protocol_sha256 is None:
        parser.error("Provide the SHA256 of the separately frozen extension protocol")
    if args.assignment_directory is None and args.phase != "assignments":
        parser.error("Provide this encoder's score-stratified assignment directory")
    original, config = original_config(REPOSITORY, 512 if args.encoder == "vit_b32" else 1024)
    inputs = original["identity"]["inputs"]
    if args.encoder == "rn50" and any(value is None for value in
            (args.train_features, args.train_metadata, args.dev_features, args.dev_metadata)):
        parser.error("RN50 requires explicit training and development caches")
    data = FeatureDataset(REPOSITORY,
        REPOSITORY / (args.train_manifest or inputs["manifest"]["path"]),
        REPOSITORY / (args.train_features or inputs["features"]["path"]),
        REPOSITORY / (args.train_metadata or inputs["metadata"]["path"]), config.feature_dim)
    output = REPOSITORY / (args.output or f"results/allocation_distillation/{args.encoder}")
    if args.phase == "assignments":
        verify_protocol(REPOSITORY, REPOSITORY / args.protocol, args.protocol_sha256, config)
        directory = REPOSITORY / args.assignment_directory if args.assignment_directory else output / "assignments"
        if not directory.resolve().is_relative_to((REPOSITORY / "results/allocation_distillation").resolve()):
            parser.error("Extension assignments must stay under results/allocation_distillation")
        records = generate_assignments(REPOSITORY, directory, data, args.protocol_sha256)
        print(json.dumps({"phase": "assignments", "assignments": records}), flush=True)
        return
    pool = SourceRetrievalPool(REPOSITORY, REPOSITORY / args.dev_manifest,
        REPOSITORY / (args.dev_features or "results/review_followup/features/e_vil_dev900/features.npz"),
        REPOSITORY / (args.dev_metadata or "results/review_followup/features/e_vil_dev900/metadata.json"),
        data, config.feature_dim)
    ledger, assignments = build_ledger(REPOSITORY, output, args.encoder, config, data, pool,
                          REPOSITORY / args.protocol, args.protocol_sha256,
                          REPOSITORY / args.assignment_directory)
    rates = (args.only_learning_rate,) if args.only_learning_rate is not None else config.learning_rates
    if any(rate not in config.learning_rates for rate in rates):
        parser.error("Learning rate is outside the frozen extension grid")
    policies, seeds = args.policies or POLICIES, args.seeds or config.seeds
    if args.phase in ("all", "select") and (args.policies or args.seeds or args.only_learning_rate is not None):
        parser.error("The all and select phases require the complete grid")
    if args.phase == "plan":
        print(json.dumps({"phase": "plan", "ledger": ledger, "full_candidates": 117,
                          "requested_candidates": len(policies) * len(seeds) * len(rates)}, indent=2), flush=True)
        return
    ensure_ledger(output, ledger)
    if args.phase in ("fit", "all"):
        for policy in policies:
            for rate in rates:
                for seed in seeds:
                    result = fit_candidate(REPOSITORY, output, config, data, pool, ledger, policy, rate, seed)
                    print(json.dumps({"phase": "fit", **{k: v for k, v in result.items()
                                                         if k != "artifact_sha256"}}), flush=True)
    if args.phase in ("select", "all"):
        manifest = select_and_export(REPOSITORY, output, config, ledger)
        print(json.dumps({"phase": "select", "state_count": len(manifest["states"]),
                          "selection_count": len(manifest["selections"])}), flush=True)
    if args.phase in ("matched-controls", "all"):
        manifest = fit_matched_controls(REPOSITORY, output, config, data, pool, ledger, assignments)
        print(json.dumps({"phase": "matched-controls", "state_count": len(manifest["states"]),
                          "selection_count": len(manifest["selections"]),
                          "complete": manifest["matched_controls_complete"]}), flush=True)


if __name__ == "__main__":
    main()
