#!/usr/bin/env python3
"""Fit frozen nonlinear ViT or linear RN50 replication and export scoring states."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcr.strengthen_replication import (
    OUTPUTS, POLICIES, PROTOCOLS, SourceRetrievalPool, build_replication_ledger,
    ensure_review_ledger, fit_replication_candidate, load_replication_context,
    prepare_assignments, select_and_export, verify_replication_protocol,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('assignments', 'plan', 'fit', 'select'))
    parser.add_argument('--setting', choices=PROTOCOLS, required=True)
    parser.add_argument('--protocol')
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output')
    parser.add_argument('--train-features')
    parser.add_argument('--train-metadata')
    parser.add_argument('--dev-features')
    parser.add_argument('--dev-metadata')
    parser.add_argument('--policies', nargs='+', choices=POLICIES)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(17, 29, 43))
    parser.add_argument('--only-learning-rate', type=float)
    args = parser.parse_args()
    protocol = ROOT / (args.protocol or PROTOCOLS[args.setting])
    output = ROOT / (args.output or OUTPUTS[args.setting])
    original, config, data = load_replication_context(ROOT, args.setting,
        ROOT / args.train_features if args.train_features else None,
        ROOT / args.train_metadata if args.train_metadata else None)
    verify_replication_protocol(protocol, args.protocol_sha256, args.setting, config, original)
    if args.phase == 'assignments':
        records = prepare_assignments(ROOT, output, args.setting, args.protocol_sha256, data)
        print(json.dumps({'phase': 'assignments', 'setting': args.setting, 'count': len(records), 'records': records}), flush=True)
        return
    feature_root = ROOT / ('results/review_followup/features' if args.setting == 'nonlinear' else 'results/strengthen_second_encoder/features/review_followup')
    pool = SourceRetrievalPool(ROOT, ROOT / 'data/review_followup/e_vil_dev900/manifest.json',
                              ROOT / args.dev_features if args.dev_features else feature_root / 'e_vil_dev900/features.npz',
                              ROOT / args.dev_metadata if args.dev_metadata else feature_root / 'e_vil_dev900/metadata.json', data, config.feature_dim)
    ledger, assignments = build_replication_ledger(ROOT, output, args.setting, protocol, args.protocol_sha256, original, config, data, pool)
    policies, seeds = args.policies or POLICIES, args.seeds or config.seeds
    rates = (args.only_learning_rate,) if args.only_learning_rate is not None else config.learning_rates
    if any(rate not in config.learning_rates for rate in rates):
        raise ValueError('Learning rate is outside frozen grid.')
    if args.phase == 'plan':
        print(json.dumps({'phase': 'plan', 'ledger': ledger, 'full_candidates': 45,
                          'requested_candidates': len(policies) * len(seeds) * len(rates)}, indent=2))
        return
    ensure_review_ledger(output, ledger)
    if args.phase == 'select':
        manifest = select_and_export(ROOT, output, config, ledger)
        print(json.dumps({'phase': 'select', 'evaluation_states': manifest['state_count'], 'selections': len(manifest['selections'])}), flush=True)
        return
    for policy in policies:
        for rate in rates:
            for seed in seeds:
                result = fit_replication_candidate(ROOT, output, config, data, pool, ledger, assignments, policy, rate, seed)
                print(json.dumps({'phase': 'fit', 'setting': args.setting, **{k: v for k, v in result.items() if k not in ('artifact_sha256', 'checkpoint_sha256')}}), flush=True)


if __name__ == '__main__':
    main()
