#!/usr/bin/env python3
"""Fit and select the reconstructed study without reading held-out scores."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gcr.training import (  # noqa: E402
    METHODS, FeatureDataset, StudyConfig, ensure_epoch_zero, ensure_ledger,
    fit_candidate, make_ledger, select_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "fit", "select"))
    parser.add_argument("--manifest", default="data/visual_entailment/manifest.json")
    parser.add_argument("--features", default="results/features/visual_entailment/features.npz")
    parser.add_argument("--feature-metadata", default="results/features/visual_entailment/metadata.json")
    parser.add_argument("--output", default="results/study")
    parser.add_argument("--protocol", default="docs/RERUN_PROTOCOL.json")
    parser.add_argument("--protocol-sha256", required=True, help="Required exact digest of the frozen reconstruction protocol.")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3], help="Whole immutable grid, never an execution filter.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="Fit execution filter; the immutable ledger still includes every method.")
    parser.add_argument("--seeds", nargs="+", type=int, choices=[17, 29, 43], help="Fit execution filter; all three seeds remain in the ledger.")
    parser.add_argument("--only-learning-rate", type=float, help="Fit execution filter within the declared grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(learning_rates=tuple(args.learning_rates), threads=args.threads)
    data = FeatureDataset(REPOSITORY, REPOSITORY / args.manifest, REPOSITORY / args.features, REPOSITORY / args.feature_metadata, config.feature_dim)
    ledger = make_ledger(REPOSITORY, data, config, REPOSITORY / args.protocol, args.protocol_sha256)
    study_root = REPOSITORY / args.output
    methods = tuple(args.methods or config.methods)
    seeds = tuple(args.seeds or config.seeds)
    rates = (args.only_learning_rate,) if args.only_learning_rate is not None else config.learning_rates
    if any(rate not in config.learning_rates for rate in rates):
        raise ValueError("Execution-filter learning rate is not in the declared grid.")
    if args.phase == "plan":
        print(json.dumps({"ledger": ledger, "requested_candidate_count": len(methods) * len(seeds) * len(rates), "full_candidate_count": len(config.methods) * len(config.seeds) * len(config.learning_rates)}, indent=2))
        return
    ensure_ledger(study_root, ledger)
    if args.phase == "select":
        selection = select_candidates(REPOSITORY, study_root, config, ledger)
        print(json.dumps({"phase": "select", "selected_checkpoint_count": selection["selected_checkpoint_count"], "selection": str(study_root / "selection.json")}), flush=True)
        return
    ensure_epoch_zero(study_root, data, config, ledger)
    for method in methods:
        for learning_rate in rates:
            for seed in seeds:
                result = fit_candidate(REPOSITORY, study_root, data, config, ledger, method, learning_rate, seed)
                print(json.dumps({"phase": "fit", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
