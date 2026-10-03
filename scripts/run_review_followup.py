#!/usr/bin/env python3
"""Freeze controls, fit all epochs, and select the independent review extension."""
import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gcr.review_training import (
    POLICIES, SourceRetrievalPool, build_review_ledger, ensure_review_ledger,
    fit_review_candidate, load_original_context, prepare_assignments,
    select_and_export, verify_protocol,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("assignments", "plan", "fit", "select"))
    parser.add_argument("--protocol", default="docs/REVIEW_FOLLOWUP_PROTOCOL.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", default="results/review_followup")
    parser.add_argument("--dev-manifest", default="data/review_followup/e_vil_dev900/manifest.json")
    parser.add_argument("--dev-features", default="results/review_followup/features/e_vil_dev900/features.npz")
    parser.add_argument("--dev-metadata", default="results/review_followup/features/e_vil_dev900/metadata.json")
    parser.add_argument("--policies", nargs="+", choices=POLICIES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(17, 29, 43))
    parser.add_argument("--only-learning-rate", type=float)
    args = parser.parse_args()
    original, config, data = load_original_context(REPOSITORY)
    protocol = REPOSITORY / args.protocol
    output = REPOSITORY / args.output
    verify_protocol(protocol, args.protocol_sha256, original, config)
    if args.phase == "assignments":
        records = prepare_assignments(REPOSITORY, output, protocol, args.protocol_sha256, original, config, data)
        print(json.dumps({"phase": "assignments", "count": len(records), "records": records}), flush=True)
        return
    pool = SourceRetrievalPool(REPOSITORY, REPOSITORY / args.dev_manifest, REPOSITORY / args.dev_features,
                               REPOSITORY / args.dev_metadata, data, config.feature_dim)
    ledger, assignments = build_review_ledger(REPOSITORY, output, protocol, args.protocol_sha256, original, config, data, pool)
    policies, seeds = args.policies or POLICIES, args.seeds or config.seeds
    rates = (args.only_learning_rate,) if args.only_learning_rate is not None else config.learning_rates
    if any(rate not in config.learning_rates for rate in rates):
        raise ValueError("Requested rate is not in the original declared grid.")
    if args.phase == "plan":
        print(json.dumps({"phase": "plan", "ledger": ledger, "full_candidates": 72,
                          "requested_candidates": len(policies) * len(seeds) * len(rates)}, indent=2))
        return
    ensure_review_ledger(output, ledger)
    if args.phase == "select":
        manifest = select_and_export(REPOSITORY, output, config, ledger)
        print(json.dumps({"phase": "select", "states": manifest["state_count"], "selections": len(manifest["selections"])}), flush=True)
        return
    for policy in policies:
        for rate in rates:
            for seed in seeds:
                result = fit_review_candidate(REPOSITORY, output, config, data, pool, ledger, assignments, policy, rate, seed)
                print(json.dumps({"phase": "fit", **{k: v for k, v in result.items() if k != "artifact_sha256"}}), flush=True)


if __name__ == "__main__":
    main()
