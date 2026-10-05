#!/usr/bin/env python3
"""Separately locked retrieval-only benchmark supplement; no new candidate gate.

Order: core lock -> supplement and LAB locks -> release -> scoring -> analysis.
The release freezes every planned state before any core benchmark outcome.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_practical_benchmark_v10 as core
from evaluate_practical_constrained_development_v8 import digest, read, record, root_path, verify_record, write_json, write_npz
from run_practical_retrieval_only_v10 import verify_contract
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer
from gcr.practical_benchmark_v10 import ENCODERS, SEEDS, DATASETS, bootstrap, paired_seed_differences, score_retrieval, score_triplets
SOURCES = ('scripts/evaluate_practical_retrieval_only_benchmark_v10.py', 'tests/test_practical_retrieval_only_benchmark_v10.py')


def verify_state(run, contract_path, contract, protocol, core_lock):
    run = root_path(run); ledger, completion = read(run / 'ledger.json'), read(run / 'completion.json')
    identity = ledger['identity']; encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != ledger['ledger_sha256'] or completion['ledger_sha256'] != ledger['ledger_sha256']:
        raise ValueError('Retrieval-only ledger identity changed')
    encoder, seed = identity['encoder'], identity['config']['seed']
    if encoder not in ENCODERS or seed not in SEEDS: raise ValueError('Unexpected retrieval-only encoder or seed')
    protocol_sha = core_lock['protocol']['sha256']
    for value in (identity, completion):
        if (value.get('study') != 'sanw_practical_v10' or value.get('family') != 'retrieval_only' or value.get('mode') != 'full'
                or value.get('encoder') != encoder or value.get('config') != {**protocol['fit_config'], 'seed': seed}
                or value.get('protocol_sha256') != protocol_sha or value.get('fixed_composition_multiplier') != 0.0
                or value.get('candidate_selection_allowed') is not False or value.get('explanatory_control_only') is not True
                or digest(verify_record(value['contract'])) != digest(root_path(contract_path))):
            raise ValueError('Retrieval-only fit differs from its fixed explanatory contract')
    if (identity['source_sha256'] != protocol['source_sha256'] or identity['control_source_sha256'] != contract['new_source_sha256']
            or digest(verify_record(identity['protocol'])) != protocol_sha or identity.get('fresh_confirmation_inclusion_allowed') is not False
            or identity.get('heldout_used_in_fitting_or_checkpoint_selection') is not False or completion.get('development_or_test_used') is not False
            or identity['fit_gallery_image_count'] != 6000 or identity['fit_gallery_text_count'] != 30000):
        raise ValueError('Retrieval-only sources or training scope differ')
    joint_state = core_lock['states'][f'{encoder}__joint__{seed}']; joint_identity = read(verify_record(joint_state['ledger']))['identity']
    if any(identity[key] != joint_identity[key] for key in ('training_provenance', 'retrieval_logit_scale', 'streaming')):
        raise ValueError('Retrieval-only data/normalization/scale/streaming provenance differs from the candidate')
    prerequisites = read(verify_record(identity['prerequisite_validation']))
    if (prerequisites.get('study') != 'sanw_practical_v10_retrieval_only_prerequisites' or prerequisites.get('passed') is not True
            or prerequisites.get('protocol_sha256') != protocol_sha or prerequisites.get('source_sha256') != contract['new_source_sha256']
            or prerequisites.get('raw_development_evidence_reconstructed') is not True or prerequisites.get('all_control_epochs_verified') is not True
            or prerequisites.get('benchmark_outcomes_read') is not False or prerequisites['development_gate'] != identity['development_gate']
            or prerequisites['completed_matched_no_retention'] != identity['completed_matched_no_retention']
            or digest(verify_record(prerequisites['contract'])) != digest(root_path(contract_path))):
        raise ValueError('Retrieval-only prerequisite receipt differs')
    if digest(verify_record(identity['development_gate']['record'])) != core_lock['development_gate']['sha256']:
        raise ValueError('Retrieval-only fit followed a different development qualification')
    expected_controls = {f'{e}__no_retention__{s}': core_lock['states'][f'{e}__no_retention__{s}'] for e in ENCODERS for s in SEEDS}
    controls = {}
    for entry in identity['completed_matched_no_retention']:
        state = entry['state']; key = f"{state['encoder']}__no_retention__{state['seed']}"
        if key in controls or digest(verify_record(entry['completion'])) != state['completion']['sha256']:
            raise ValueError('Prerequisite matched controls are duplicated or changed')
        controls[key] = state
    if controls != expected_controls: raise ValueError('Retrieval-only fit did not follow the six locked matched controls')
    rows, history = completion['checkpoint_history'], completion['history']
    if (read(run / 'history.json') != rows or [{k: v for k, v in r.items() if k != 'checkpoint'} for r in rows] != history
            or [r['epoch'] for r in history] != list(range(1, 33)) or completion['optimizer_steps'] != 3008
            or any(r['optimizer_steps'] != r['epoch'] * 94 or r['fixed_composition_multiplier'] != 0.0
                   or not np.isfinite(r['training_objective']) or not all(r['certificate'].get(k) is True for k in
                   ('ranking_checked_canonically', 'ranking_preserved', 'feasible_with_tolerance')) for r in history)):
        raise ValueError('Retrieval-only training budget, certificate or zero multiplier changed')
    chosen = min((r for r in history if r['nonzero']), key=lambda r: (r['training_objective'], r['epoch']))
    if (completion['selection'] != 'minimum_feasible_nonzero_training_objective_then_earliest_epoch'
            or chosen['epoch'] != completion['selected_epoch'] or chosen['training_objective'] != completion['selected_training_objective']):
        raise ValueError('Retrieval-only checkpoint was not training-objective-selected')
    if not all(completion['final_certificate'].get(k) is True for k in ('ranking_checked_canonically', 'ranking_preserved', 'feasible_with_tolerance')):
        raise ValueError('Retrieval-only final finite-training certificate missing')
    checkpoint = run / completion['selected_checkpoint']['path']
    if digest(checkpoint) != completion['selected_checkpoint']['sha256']: raise ValueError('Selected checkpoint changed')
    selected = ConstrainedBilinearScorer.load(checkpoint); joint = ConstrainedBilinearScorer.load(verify_record(joint_state['checkpoint']))
    geometry = ('image_mean', 'text_mean', 'image_basis', 'text_basis')
    if (selected.coefficient.shape != (128, 128) or len(selected.image_mean) != (512 if encoder == 'vit_b32' else 1024)
            or any(not np.array_equal(getattr(selected, k), getattr(joint, k)) for k in geometry)):
        raise ValueError('Retrieval-only geometry differs from matched candidate')
    for row in rows:
        entry = row['checkpoint']; path = run / entry['path']
        if digest(path) != entry['sha256'] or entry['ledger_sha256'] != ledger['ledger_sha256']: raise ValueError('Epoch bytes/ledger changed')
        model = ConstrainedBilinearScorer.load(path); norm2 = float(np.sum(model.coefficient * model.coefficient, dtype=np.float64))
        if (any(not np.array_equal(getattr(model, k), getattr(selected, k)) for k in geometry)
                or row['nonzero'] != bool(np.any(model.coefficient != 0)) or not np.isfinite(norm2)
                or norm2 > protocol['fit_config']['radius'] ** 2 * (1 + 128 * np.finfo(np.float64).eps)):
            raise ValueError('Epoch geometry, nonzero flag or radius differs')
        if row['epoch'] == chosen['epoch'] and not np.array_equal(model.coefficient, selected.coefficient): raise ValueError('Selected state differs from chosen epoch')
    return {'run': str(run.relative_to(ROOT)), 'encoder': encoder, 'family': 'retrieval_only', 'seed': seed, 'epoch': chosen['epoch'],
            'nonzero': True, 'ledger': record(run / 'ledger.json'), 'completion': record(run / 'completion.json'), 'checkpoint': record(checkpoint)}


def build_lock(core_path, core_sha, contract_path, source_audit_path, runs):
    core_lock = core.load_lock(core_path, core_sha); contract, protocol, _ = verify_contract(ROOT, root_path(contract_path))
    if contract['inherited_protocol']['sha256'] != core_lock['protocol']['sha256']: raise ValueError('Supplement and core protocols differ')
    sources = {n: digest(ROOT / n) for n in SOURCES}; audit = read(source_audit_path)
    if (audit.get('study') != 'sanw_practical_v10_independent_retrieval_only_benchmark_source_audit' or audit.get('passed') is not True
            or audit.get('blocking_findings') != [] or audit['contract']['sha256'] != digest(root_path(contract_path))
            or any(audit.get('sources', {}).get(n) != v for n, v in sources.items())): raise ValueError('Independent supplement source review required')
    states = {}
    for run in runs:
        state = verify_state(run, contract_path, contract, protocol, core_lock); key = f"{state['encoder']}__{state['seed']}"
        if key in states: raise ValueError('Duplicate retrieval-only state')
        states[key] = state
    if set(states) != {f'{e}__{s}' for e in ENCODERS for s in SEEDS}: raise ValueError('All six fixed retrieval-only states required')
    return {'study': 'sanw_practical_v10_retrieval_only_benchmark_lock', 'core_lock': record(core_path), 'contract': record(contract_path),
            'source_audit': record(source_audit_path), 'sources': sources, 'states': states, 'inputs': core_lock['inputs'],
            'contrasts': contract['contrasts'], 'endpoints': contract['endpoints'], 'inference': contract['inference'], 'effect_count': 20,
            'candidate_selection_allowed': False, 'fresh_confirmation_inclusion_allowed': False,
            'core_result_conditioning': 'report_every_supplement_effect_regardless_core_practical_gate', 'evaluation_status': core_lock['evaluation_status']}


def load_lock(path, sha):
    if digest(root_path(path)) != sha: raise ValueError('Supplement lock SHA mismatch')
    lock = read(path); rebuilt = build_lock(lock['core_lock']['path'], lock['core_lock']['sha256'], lock['contract']['path'],
                                           lock['source_audit']['path'], [s['run'] for s in lock['states'].values()])
    if lock != rebuilt: raise ValueError('Supplement source/state/contract/input changed')
    return lock


def verify_additional_lock(path, core_entry):
    value = read(path)
    if (value.get('study') != 'sanw_practical_v10_additional_benchmark_lock' or value.get('family') != 'labclip'
            or value.get('core_lock') != core_entry or value.get('candidate_selection_allowed') is not False
            or value.get('fresh_confirmation_inclusion_allowed') is not False or value.get('effect_count') != 10):
        raise ValueError('Additional lock must be the audited ten-effect LABCLIP comparison')
    core_lock = read(verify_record(core_entry))
    if value.get('protocol_sha256') != core_lock['protocol']['sha256']: raise ValueError('Additional protocol differs')
    verify_record(value['contract']); audit = read(verify_record(value['source_audit']))
    if (audit.get('passed') is not True or audit.get('blocking_findings') != [] or audit.get('contract', {}).get('sha256') != value['contract']['sha256']
            or not value.get('sources') or any(audit.get('sources', {}).get(n) != sha for n, sha in value['sources'].items())):
        raise ValueError('Additional family lacks matching source audit')
    for name, sha in value['sources'].items():
        if digest(ROOT / name) != sha: raise ValueError('Additional source changed')
    if set(value['states']) != {f'{e}__{s}' for e in ENCODERS for s in SEEDS}: raise ValueError('Additional family requires fixed six states')
    for key, state in value['states'].items():
        if key != f"{state['encoder']}__{state['seed']}" or state.get('family') != 'labclip' or state.get('nonzero') is not True:
            raise ValueError('Additional state family/nonzero differs')
        for field in ('ledger', 'completion', 'checkpoint'): verify_record(state[field])
    if 'scripts/evaluate_practical_labclip_benchmark_v10.py' not in value['sources']: raise ValueError('Missing authoritative additional validator binding')
    from evaluate_practical_labclip_benchmark_v10 import load_lock as load_labclip_lock
    if load_labclip_lock(path, digest(root_path(path))) != value: raise ValueError('Authoritative LABCLIP lock reconstruction differs')
    return value


def verify_output_roots(core_root, supplement_root, additional_roots=(), *, empty=False):
    roots = [root_path(core_root), root_path(supplement_root), *(root_path(p) for p in additional_roots)]
    for i, path in enumerate(roots):
        if any(path == other or path.is_relative_to(other) or other.is_relative_to(path) for other in roots[:i]):
            raise ValueError('Output roots must be distinct and nonnested')
        if empty and path.exists() and any(path.iterdir()): raise ValueError('Release must precede every benchmark output')
    return roots


def release(args, lock):
    additional, families, extra_roots = [], set(), []
    for path in args.additional_locks:
        value = verify_additional_lock(path, lock['core_lock'])
        if value['family'] in families: raise ValueError('Duplicate additional explanatory family')
        families.add(value['family']); additional.append(path); extra_roots.append(value['output_root'])
    roots = verify_output_roots(args.core_output_root, args.supplement_output_root, extra_roots, empty=True)
    additional = [record(path) for path in additional]
    value = {'study': 'sanw_practical_v10_full_benchmark_release', 'supplement_lock': record(args.lock), 'core_lock': lock['core_lock'],
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'core_output_root': str(roots[0].relative_to(ROOT)),
             'supplement_output_root': str(roots[1].relative_to(ROOT)), 'additional_locks': additional,
             'selected_states_frozen_before_scoring': 18 + 6 * len(additional), 'core_source_unchanged': True,
             'supplement_reported_regardless_core_gate': True, 'fresh_confirmation_included': False,
             'execution_order': ['lock_core', 'lock_supplement', 'release', 'score-core_and_score', 'analyze_core_and_supplement']}
    print(json.dumps(write_json(args.output, value)), flush=True)


def load_release(path, sha):
    if digest(root_path(path)) != sha: raise ValueError('Benchmark release SHA mismatch')
    value = read(path); lock = load_lock(value['supplement_lock']['path'], value['supplement_lock']['sha256'])
    additional, families, roots = value.get('additional_locks', []), set(), []
    for entry in additional:
        extra = verify_additional_lock(verify_record(entry), lock['core_lock'])
        if extra['family'] in families: raise ValueError('Duplicate additional explanatory family')
        families.add(extra['family']); roots.append(extra['output_root'])
    verify_output_roots(value['core_output_root'], value['supplement_output_root'], roots)
    if (value.get('study') != 'sanw_practical_v10_full_benchmark_release' or value['core_lock'] != lock['core_lock']
            or value.get('selected_states_frozen_before_scoring') != 18 + 6 * len(additional)
            or value.get('fresh_confirmation_included') is not False): raise ValueError('Incomplete final release')
    return value, lock


def output_for(release_record, operation, encoder, dataset):
    if encoder not in ENCODERS or dataset not in DATASETS or operation not in ('core', 'supplement'): raise ValueError('Unexpected scope')
    return root_path(release_record[f'{operation}_output_root']) / encoder / dataset


def score(args, release_record, lock):
    operation = 'core' if args.mode == 'score-core' else 'supplement'; output = output_for(release_record, operation, args.encoder, args.dataset)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing to adopt existing outcomes')
    core_lock = read(verify_record(lock['core_lock']))
    if operation == 'core': core.evaluate(SimpleNamespace(output=str(output), encoder=args.encoder, dataset=args.dataset, lock=lock['core_lock']['path']), core_lock)
    else:
        protocol = read(verify_record(core_lock['protocol'])); config = {'datasets': {n: {k: e['path'] for k, e in v.items()} for n, v in lock['inputs'][args.encoder].items()}}
        metadata = read(verify_record(protocol['original_training_inputs'][args.encoder]['metadata'])); dataset = core.load_dataset(config, args.dataset, args.encoder, metadata)
        output.mkdir(parents=True, exist_ok=True)
        start = write_json(output / 'prescore_receipt.json', {'release': record(args.release), 'supplement_lock': release_record['supplement_lock'],
                           'core_lock': lock['core_lock'], 'encoder': args.encoder, 'dataset': args.dataset,
                           'inputs': lock['inputs'][args.encoder][args.dataset], 'canonical_score_shared_with_core': True})
        artifacts = {}
        for seed in SEEDS:
            state = lock['states'][f'{args.encoder}__{seed}']; m = ConstrainedBilinearScorer.load(verify_record(state['checkpoint']))
            scorer = CanonicalScorer(m.image_mean, m.text_mean, m.image_basis, m.text_basis, m.coefficient)
            summary, raw = score_triplets(dataset, scorer) if args.dataset == 'sugarcrepe_pp' else score_retrieval(dataset, scorer)
            artifacts[f'retrieval_only_{seed}'] = {'state': state, 'summary': summary, 'predictions': write_npz(output / f'retrieval_only_{seed}.npz', raw)}
        for entry in lock['inputs'][args.encoder][args.dataset].values(): verify_record(entry)
        write_json(output / 'index.json', {'study': 'sanw_practical_v10_retrieval_only_benchmark_predictions', 'status': 'complete',
                   'release': record(args.release), 'encoder': args.encoder, 'dataset': args.dataset, 'start_receipt': start, 'artifacts': artifacts})
    write_json(output / 'release_run_receipt.json', {'study': 'sanw_practical_v10_released_benchmark_run', 'operation': operation,
               'release': record(args.release), 'encoder': args.encoder, 'dataset': args.dataset, 'index': record(output / 'index.json')})
    print(json.dumps({'index': record(output / 'index.json'), 'release_run_receipt': record(output / 'release_run_receipt.json')}), flush=True)


def load_prediction_index(release_record, release_path, lock, operation, encoder, dataset):
    output = output_for(release_record, operation, encoder, dataset); receipt = read(output / 'release_run_receipt.json')
    if (receipt.get('study') != 'sanw_practical_v10_released_benchmark_run' or receipt.get('operation') != operation
            or receipt['release'] != record(release_path) or receipt['encoder'] != encoder or receipt['dataset'] != dataset
            or verify_record(receipt['index']) != output / 'index.json'): raise ValueError('Index not bound to prescoring release')
    index = read(output / 'index.json'); start = read(verify_record(index['start_receipt'])); core_lock = read(verify_record(lock['core_lock']))
    if index['encoder'] != encoder or index['dataset'] != dataset or index.get('status') != 'complete': raise ValueError('Index scope incomplete')
    if operation == 'core':
        if (index['study'] != 'sanw_practical_v10_benchmark_predictions' or index['lock'] != lock['core_lock']
                or start['lock'] != lock['core_lock'] or start['encoder'] != encoder or start['dataset'] != dataset): raise ValueError('Core prediction identity differs')
        expected = {'frozen': None} | {f'{f}_{s}': core_lock['states'][f'{encoder}__{f}__{s}'] for f in ('joint', 'no_retention') for s in SEEDS}
    else:
        if (index['study'] != 'sanw_practical_v10_retrieval_only_benchmark_predictions' or index['release'] != record(release_path)
                or start['release'] != record(release_path) or start['core_lock'] != lock['core_lock'] or start['encoder'] != encoder or start['dataset'] != dataset
                or start['supplement_lock'] != release_record['supplement_lock'] or start['inputs'] != lock['inputs'][encoder][dataset]
                or start['canonical_score_shared_with_core'] is not True): raise ValueError('Supplement prediction identity differs')
        expected = {f'retrieval_only_{s}': lock['states'][f'{encoder}__{s}'] for s in SEEDS}
    if set(index['artifacts']) != set(expected): raise ValueError('Incomplete fixed-seed prediction coverage')
    raw = {}
    for name, item in index['artifacts'].items():
        if item['state'] != expected[name]: raise ValueError('Prediction state differs')
        verify_record(item['predictions'])
        if not name.startswith('no_retention_'): raw[name] = core.archive(item['predictions'])
    return raw, record(output / 'index.json'), record(output / 'release_run_receipt.json')


def analyze(args, release_record, lock):
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing to overwrite effects')
    output.mkdir(parents=True, exist_ok=True); (output / 'paired').mkdir(); effects, draws, paired, indices = [], {}, {}, []
    for encoder in ENCODERS:
        for dataset in DATASETS:
            raw = {}
            for operation in ('core', 'supplement'):
                values, index, receipt = load_prediction_index(release_record, args.release, lock, operation, encoder, dataset)
                raw.update(values); indices.append({'index': index, 'release_run_receipt': receipt})
            for contrast, first, second in lock['contrasts']:
                for metric in lock['endpoints'][dataset]:
                    selected = lambda f: {s: raw['frozen' if f == 'frozen' else f'{f}_{s}'] for s in SEEDS}
                    delta, clusters = paired_seed_differences(dataset, metric, selected(first), selected(second)); effect, samples = bootstrap(delta, clusters)
                    key = f"{encoder}__{dataset}__{metric.replace('.', '_')}__{contrast}"
                    effects.append({**effect, 'effect_id': key, 'encoder': encoder, 'dataset': dataset, 'metric': metric, 'contrast': contrast}); draws[key] = samples
                    paired[key] = write_npz(output / 'paired' / f'{key}.npz', {'difference_by_seed': delta, 'cluster_ids': clusters, 'fixed_seeds': np.asarray(SEEDS)})
    if len(effects) != 20: raise ValueError('Exactly all20 supplementary effects required')
    print(json.dumps(write_json(output / 'analysis.json', {'study': 'sanw_practical_v10_retrieval_only_benchmark_analysis', 'release': record(args.release),
          'supplement_lock': release_record['supplement_lock'], 'core_lock': lock['core_lock'], 'indices': indices, 'effects': effects,
          'paired_artifacts': paired, 'bootstrap': write_npz(output / 'bootstrap.npz', draws), 'effect_count': 20,
          'core_practical_result_used_to_filter_effects': False, 'candidate_selection_allowed': False, 'fresh_confirmation_inclusion_allowed': False,
          'new_gate': None, 'evaluation_status': lock['evaluation_status']})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True); q = sub.add_parser('lock')
    for n in ('core-lock', 'core-lock-sha256', 'contract', 'source-audit', 'output'): q.add_argument('--' + n, required=True)
    q.add_argument('--runs', nargs=6, required=True); q = sub.add_parser('release')
    for n in ('lock', 'lock-sha256', 'core-output-root', 'supplement-output-root', 'output'): q.add_argument('--' + n, required=True)
    q.add_argument('--additional-locks', nargs='*', default=[])
    for n in ('score-core', 'score', 'analyze'):
        q = sub.add_parser(n); q.add_argument('--release', required=True); q.add_argument('--release-sha256', required=True)
        if n == 'analyze': q.add_argument('--output', required=True)
        else: q.add_argument('--encoder', choices=ENCODERS, required=True); q.add_argument('--dataset', choices=DATASETS, required=True)
    args = p.parse_args(); core.development.require_evaluation_threads()
    if args.mode == 'lock': print(json.dumps(write_json(args.output, build_lock(args.core_lock, args.core_lock_sha256, args.contract, args.source_audit, args.runs))), flush=True)
    elif args.mode == 'release': release(args, load_lock(args.lock, args.lock_sha256))
    else:
        release_record, lock = load_release(args.release, args.release_sha256); (analyze if args.mode == 'analyze' else score)(args, release_record, lock)
if __name__ == '__main__': main()
