#!/usr/bin/env python3
"""Separate fixed ten-effect equal-data LABCLIP comparison; no new gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_practical_benchmark_v10 as core
import evaluate_practical_retrieval_only_benchmark_v10 as release_api
import run_practical_labclip_v10 as fit_api
from evaluate_practical_constrained_development_v8 import digest, read, record, root_path, verify_record, write_json, write_npz
from gcr.practical_inner_evaluation_v9 import LABCLIPScorer
from gcr.practical_labclip_v10 import select_epoch
from gcr.practical_benchmark_v10 import ENCODERS, SEEDS, DATASETS, score_retrieval, score_triplets, paired_seed_differences, bootstrap
SOURCES = ('scripts/evaluate_practical_labclip_benchmark_v10.py', 'tests/test_practical_labclip_benchmark_v10.py') + release_api.SOURCES
AUDIT_STUDY = 'sanw_practical_v10_independent_labclip_benchmark_source_audit'
WITNESS_FIELDS = ('source_rows', 'owner_rows', 'pair_count', 'tolerance', 'max_direction_change_from_identity', 'max_owned_pair_score_change_from_identity', 'nonzero_functional_update')
AUDIT_SCOPE = 'all32_objective_minimum_source_and_checkpoint_verified;identity_and_selected_movement_and_selected_objective_numerically_replayed'
SCORE_IDENTITY = 'bound_v9_LABCLIPScorer_canonical_transform_then_shared_fixed_pair_sum'


def load_model(path, dimension, native_scale):
    with np.load(path, allow_pickle=False) as a:
        if (set(a.files) != {'schema', 'weight', 'log_scale', 'learned_scale', 'score'} or str(a['schema']) != 'sanw_labclip_inner_v9'
                or str(a['score']) != 'image_dot_l2_normalized_full_rank_transformed_text' or a['learned_scale'].shape != () or bool(a['learned_scale'])
                or a['weight'].dtype != np.float32 or a['weight'].shape != (dimension, dimension) or not np.isfinite(a['weight']).all()
                or a['log_scale'].shape != () or float(a['log_scale']) != math.log(native_scale)):
            raise ValueError('LABCLIP checkpoint schema, fixed scale, dtype or shape differs')
        return a['weight'].copy()


def verify_state(run, contract_path, contract, protocol, core_lock):
    run = root_path(run); ledger, value = read(run / 'ledger.json'), read(run / 'completion.json'); identity = ledger['identity']
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != ledger['ledger_sha256'] or value['ledger_sha256'] != ledger['ledger_sha256']: raise ValueError('LABCLIP ledger changed')
    encoder, seed = identity['encoder'], identity['config']['seed']; protocol_sha = core_lock['protocol']['sha256']
    if encoder not in ENCODERS or seed not in SEEDS: raise ValueError('Unexpected LABCLIP encoder/seed')
    for item in (identity, value):
        if (item.get('study') != fit_api.STUDY or item.get('family') != 'labclip' or item.get('encoder') != encoder
                or item.get('config') != {**contract['config'], 'seed': seed} or item.get('protocol_sha256') != protocol_sha
                or digest(verify_record(item['contract'])) != digest(root_path(contract_path)) or item.get('candidate_selection_allowed') is not False
                or item.get('fresh_confirmation_inclusion_allowed') is not False or item.get('heldout_used_in_fitting_or_selection') is not False):
            raise ValueError('LABCLIP fit identity differs')
    joint = core_lock['states'][f'{encoder}__joint__{seed}']; joint_identity = read(verify_record(joint['ledger']))['identity']
    if (identity['training_provenance'] != joint_identity['training_provenance'] or identity['source_sha256'] != protocol['source_sha256']
            or identity['new_source_sha256'] != contract['new_source_sha256'] or identity['source_audit'] != contract['source_audit']
            or digest(verify_record(identity['protocol'])) != protocol_sha or identity['selection'] != contract['selection']
            or identity['supported_captions_used_as_loss_targets'] is not False or identity['supported_labels_used_for_negative_conflict_exclusion'] is not True):
        raise ValueError('LABCLIP data/source/supervision identity differs')
    pre = read(verify_record(identity['prerequisite_validation']))
    if (pre.get('study') != fit_api.STUDY + '_prerequisites' or pre.get('passed') is not True or pre.get('protocol_sha256') != protocol_sha
            or pre.get('benchmark_outcomes_read') is not False or digest(verify_record(pre['contract'])) != digest(root_path(contract_path))
            or digest(verify_record(pre['development_gate']['record'])) != core_lock['development_gate']['sha256']
            or {f"{s['encoder']}__joint__{s['seed']}": s for s in pre['development_gate']['qualified_joint_states']}
               != {k: s for k, s in core_lock['states'].items() if s['family'] == 'joint'}): raise ValueError('LABCLIP prerequisite differs')
    eligible, omitted = identity['eligible_owner_indices'], identity['omitted_no_valid_contradiction_owner_indices']; count = len(eligible); steps = 5 * ((count + 127) // 128)
    if (not count or eligible != sorted(set(eligible)) or omitted != sorted(set(omitted)) or set(eligible) & set(omitted)
            or set(eligible) | set(omitted) != set(range(6000)) or identity['requested_owner_count'] != 6000
            or identity['eligible_owner_count'] != count or value['eligible_owner_count'] != count or value['omitted_no_valid_contradiction_owner_indices'] != omitted
            or identity['eligible_source_caption_count'] != 5 * count or identity['source_rows_per_epoch'] != 5 * count
            or identity['updates_per_epoch'] != steps or identity['optimizer_update_budget'] != 32 * steps or value['optimizer_steps'] != 32 * steps):
        raise ValueError('LABCLIP eligibility partition or budget differs')
    scale = float(read(verify_record(protocol['training_inputs'][encoder]['metadata']))['logit_scale'])
    if identity['native_logit_scale'] != scale: raise ValueError('LABCLIP native fixed scale differs')
    rows, history, bank = value['checkpoint_history'], value['history'], value['audit_bank']
    if (read(run / 'checkpoint_history.json') != rows or [r['epoch'] for r in rows] != list(range(33)) or [r['epoch'] for r in history] != list(range(1, 33))
            or bank['seed'] != contract['selection']['audit_seed'] or bank['batch_count'] != steps or bank['source_rows'] != 5 * count
            or value['checkpoint_candidates'] != list(range(1, 33)) or value['selection'] != contract['selection']['rule']): raise ValueError('LABCLIP fixed checkpoint/audit coverage differs')
    chosen = select_epoch(rows)
    if chosen != value['selected_epoch'] or value.get('nonzero') is not True: raise ValueError('LABCLIP selected epoch not fixed training minimum')
    dimension = 512 if encoder == 'vit_b32' else 1024; identity_path, selected_path, witness_rows = None, None, None
    for row in rows:
        epoch = row['epoch']; entry = row['checkpoint']; path = run / entry['path']
        if (digest(path) != entry['sha256'] or entry['ledger_sha256'] != ledger['ledger_sha256'] or row['optimizer_steps'] != steps * epoch
                or not np.isfinite(row['training_objective']) or row['audit_bank_sha256'] != bank['sha256']): raise ValueError('LABCLIP epoch identity or budget differs')
        if epoch and (any(row.get(k) != v for k, v in history[epoch - 1].items()) or row['batches'] != steps or row['caption_rows'] != 5 * count
                      or row['repeated_image_rows_within_batches'] != 0): raise ValueError('LABCLIP online history differs')
        weight = load_model(path, dimension, scale); w = row['functional_witness']; tolerance = contract['selection']['witness_tolerance']
        movement = w['max_direction_change_from_identity'] > tolerance and w['max_owned_pair_score_change_from_identity'] > tolerance
        if (w['tolerance'] != tolerance or w['pair_count'] != 128 or len(w['source_rows']) != 128 or len(w['owner_rows']) != 128
                or any(owner not in eligible for owner in w['owner_rows']) or w['source_rows'] != sorted(set(w['source_rows']))
                or not all(np.isfinite(w[k]) and w[k] >= 0 for k in ('max_direction_change_from_identity', 'max_owned_pair_score_change_from_identity'))
                or row['nonzero_functional_update'] != movement or w['nonzero_functional_update'] != movement): raise ValueError('LABCLIP movement witness malformed')
        current = (w['source_rows'], w['owner_rows'])
        if witness_rows is None: witness_rows = current
        if current != witness_rows: raise ValueError('LABCLIP witness rows changed between epochs')
        if epoch == 0:
            if not np.array_equal(weight, np.eye(dimension, dtype=np.float32)) or movement: raise ValueError('LABCLIP identity state differs')
            identity_path = path
        if epoch == chosen:
            if value['selected_checkpoint'] != entry or not movement or np.array_equal(weight, np.eye(dimension, dtype=np.float32)): raise ValueError('LABCLIP selected bytes/nonzero differ')
            selected_path = path
    return {'run': str(run.relative_to(ROOT)), 'encoder': encoder, 'family': 'labclip', 'seed': seed, 'epoch': chosen, 'nonzero': True,
            'native_logit_scale': scale, 'ledger': record(run / 'ledger.json'), 'completion': record(run / 'completion.json'),
            'checkpoint': record(selected_path), 'identity_checkpoint': record(identity_path)}


def audit_training(args):
    """Later training-only replay; remaining31 objectives are not numerically replayed."""
    output = root_path(args.output)
    if output.exists(): raise FileExistsError('Training audit already exists')
    contract, protocol, protocol_path = fit_api.verify_contract(ROOT, root_path(args.contract)); core_lock = core.load_lock(args.core_lock, args.core_lock_sha256)
    state = verify_state(args.run, args.contract, contract, protocol, core_lock)
    from gcr.practical_training_data_v10 import load_training
    from gcr.practical_labclip_v10 import fixed_audit_batches, audit_bank_record, fixed_training_objective, functional_witness
    from gcr.practical_labclip_v9 import NormalizedFullRankAlignment
    import torch
    torch.set_num_threads(1)
    images, texts, source_rows, owner, sources, supported, negatives, provenance, scale = load_training(ROOT, state['encoder'], protocol)
    identity = read(verify_record(state['ledger']))['identity']; completion = read(verify_record(state['completion']))
    eligible = [i for i, r in enumerate(negatives) if r]; omitted = [i for i, r in enumerate(negatives) if not r]
    if (eligible != identity['eligible_owner_indices'] or omitted != identity['omitted_no_valid_contradiction_owner_indices']
            or provenance != identity['training_provenance'] or scale != identity['native_logit_scale']): raise ValueError('Reconstructed training eligibility/provenance differs')
    batches = fixed_audit_batches(np.asarray(eligible, dtype=np.int64), sources, negatives); bank = audit_bank_record(batches)
    if bank != completion['audit_bank']: raise ValueError('Training audit bank does not reconstruct')
    rows = {r['epoch']: r for r in completion['checkpoint_history']}; movement = {}
    for name, epoch in (('identity', 0), ('selected', state['epoch'])):
        weight = load_model(verify_record(state['identity_checkpoint' if name == 'identity' else 'checkpoint']), images.shape[1], scale)
        w = functional_witness(images, texts, source_rows, owner, np.asarray(eligible), weight)
        if any(w[k] != rows[epoch]['functional_witness'][k] for k in WITNESS_FIELDS): raise ValueError('Canonical witness does not reconstruct')
        movement[name] = {k: w[k] for k in WITNESS_FIELDS}
    torch.set_num_threads(3)
    model = NormalizedFullRankAlignment(images.shape[1], native_logit_scale=scale, learned_scale=False)
    with torch.no_grad(): model.linear.weight.copy_(torch.from_numpy(load_model(verify_record(state['checkpoint']), images.shape[1], scale)))
    replay = fixed_training_objective(model, torch.tensor(images, dtype=torch.float32), torch.tensor(texts, dtype=torch.float32), batches)
    recorded = rows[state['epoch']]['training_objective']
    if replay != recorded: raise ValueError('Selected fixed training objective did not replay exactly')
    value = {'study': 'sanw_practical_v10_labclip_completed_training_audit', 'passed': True, 'core_lock': record(args.core_lock),
             'contract': record(args.contract), 'protocol_sha256': digest(protocol_path), 'state': state,
             'source_sha256': {n: digest(ROOT / n) for n in SOURCES}, 'training_provenance': provenance,
             'eligible_owner_indices': eligible, 'omitted_owner_indices': omitted, 'audit_bank': bank, 'movement': movement,
             'selected_training_objective_recorded': recorded, 'selected_training_objective_replayed': replay, 'objective_torch_threads': 3,
             'all_checkpoint_objectives_numerically_replayed': False, 'audit_scope': AUDIT_SCOPE, 'all32_checkpoint_hashes_verified': True,
             'all32_recorded_objective_minimum_verified': True, 'heldout_used_in_numerical_replay': False, 'benchmark_outcomes_read': False}
    print(json.dumps(write_json(output, value)), flush=True)


def verify_training_audit(path, state, contract_path, core_path):
    value = read(path); completion = read(verify_record(state['completion'])); identity = read(verify_record(state['ledger']))['identity']
    rows = {r['epoch']: r for r in completion['checkpoint_history']}
    if (value.get('study') != 'sanw_practical_v10_labclip_completed_training_audit' or value.get('passed') is not True or value.get('state') != state
            or value.get('contract') != record(contract_path) or value.get('core_lock') != record(core_path) or value.get('protocol_sha256') != identity['protocol_sha256']
            or value.get('source_sha256') != {n: digest(ROOT / n) for n in SOURCES} or value.get('training_provenance') != identity['training_provenance']
            or value.get('eligible_owner_indices') != identity['eligible_owner_indices'] or value.get('omitted_owner_indices') != identity['omitted_no_valid_contradiction_owner_indices']
            or value.get('audit_bank') != completion['audit_bank'] or value.get('objective_torch_threads') != 3
            or value.get('selected_training_objective_recorded') != rows[state['epoch']]['training_objective']
            or value.get('selected_training_objective_replayed') != rows[state['epoch']]['training_objective'] or value.get('audit_scope') != AUDIT_SCOPE
            or value.get('all_checkpoint_objectives_numerically_replayed') is not False or value.get('all32_checkpoint_hashes_verified') is not True
            or value.get('all32_recorded_objective_minimum_verified') is not True or value.get('heldout_used_in_numerical_replay') is not False
            or value.get('benchmark_outcomes_read') is not False): raise ValueError('Required numerical training audit missing or inconsistent')
    for name, epoch in (('identity', 0), ('selected', state['epoch'])):
        if value['movement'][name] != {k: rows[epoch]['functional_witness'][k] for k in WITNESS_FIELDS}: raise ValueError('Replayed witness differs')
    if value['movement']['identity']['nonzero_functional_update'] or not value['movement']['selected']['nonzero_functional_update']: raise ValueError('Replayed identity/selected movement differs')
    return record(path)


def build_lock(core_path, core_sha, contract_path, audit_path, runs, output_root, *, training_audits, require_empty=True):
    core_lock = core.load_lock(core_path, core_sha); contract, protocol, _ = fit_api.verify_contract(ROOT, root_path(contract_path))
    if contract['inherited_protocol']['sha256'] != core_lock['protocol']['sha256']: raise ValueError('LABCLIP/core protocols differ')
    sources = {n: digest(ROOT / n) for n in SOURCES}; audit = read(audit_path)
    if (audit.get('study') != AUDIT_STUDY or audit.get('passed') is not True or audit.get('blocking_findings') != []
            or audit['contract']['sha256'] != digest(root_path(contract_path)) or any(audit.get('sources', {}).get(n) != sha for n, sha in sources.items())):
        raise ValueError('LABCLIP evaluator requires independent source review')
    output_root = root_path(output_root)
    if require_empty and output_root.exists() and any(output_root.iterdir()): raise ValueError('LABCLIP lock must precede outcomes')
    if len(training_audits) != len(runs): raise ValueError('Each LABCLIP fit needs its completed numerical audit')
    states, numerical = {}, {}
    for run, training_audit in zip(runs, training_audits):
        state = verify_state(run, contract_path, contract, protocol, core_lock); key = f"{state['encoder']}__{state['seed']}"
        if key in states: raise ValueError('Duplicate LABCLIP seed state')
        states[key] = state; numerical[key] = verify_training_audit(training_audit, state, contract_path, core_path)
    if set(states) != {f'{e}__{s}' for e in ENCODERS for s in SEEDS}: raise ValueError('All six LABCLIP states must freeze together')
    return {'study': 'sanw_practical_v10_additional_benchmark_lock', 'family': 'labclip', 'core_lock': record(core_path),
            'protocol_sha256': core_lock['protocol']['sha256'], 'contract': record(contract_path), 'source_audit': record(audit_path),
            'sources': sources, 'states': states, 'training_audits': numerical, 'training_audit_scope': AUDIT_SCOPE, 'inputs': core_lock['inputs'],
            'output_root': str(output_root.relative_to(ROOT)), 'contrasts': contract['contrasts'], 'endpoints': contract['endpoints'],
            'inference': contract['inference'], 'effect_count': 10, 'candidate_selection_allowed': False,
            'fresh_confirmation_inclusion_allowed': False, 'evaluation_status': core_lock['evaluation_status']}


def load_lock(path, sha):
    if digest(root_path(path)) != sha: raise ValueError('LABCLIP lock SHA mismatch')
    lock = read(path)
    rebuilt = build_lock(lock['core_lock']['path'], lock['core_lock']['sha256'], lock['contract']['path'], lock['source_audit']['path'],
                         [s['run'] for s in lock['states'].values()], lock['output_root'],
                         training_audits=[verify_record(lock['training_audits'][k]) for k in lock['states']], require_empty=False)
    if rebuilt != lock: raise ValueError('LABCLIP source/state/input lock changed')
    return lock


def load_released(args):
    release, supplement = release_api.load_release(args.release, args.release_sha256)
    locks = [e for e in release['additional_locks'] if read(verify_record(e)).get('family') == 'labclip']
    if len(locks) != 1 or release['selected_states_frozen_before_scoring'] != 24: raise ValueError('All24 planned states must be released')
    lock = load_lock(locks[0]['path'], locks[0]['sha256'])
    if lock['core_lock'] != release['core_lock']: raise ValueError('LABCLIP/core lock differs')
    return release, supplement, lock, locks[0]


def identity_parity(dataset, frozen, identity):
    result = {'descriptive_only': True, 'reference': 'original_frozen', 'comparison': 'identity_normalized_text_map'}
    if dataset['manifest'].get('triplets'):
        result['correctness_changes'] = int(np.count_nonzero(frozen['correct'] != identity['correct']))
        result['max_absolute_score_error_all_triplet_pairs'] = max(float(np.max(np.abs(frozen[f'{k}_scores'] - identity[f'{k}_scores']))) for k in ('positive1', 'positive2', 'negative'))
        return result
    images = np.asarray(dataset['arrays']['image_features'], dtype=np.float64); texts = np.asarray(dataset['arrays']['text_features'], dtype=np.float64)
    normalized = LABCLIPScorer(np.eye(texts.shape[1])).transform(texts); change = normalized - texts
    image_norm = float(np.sqrt(np.sum(images * images, axis=1)).max())
    guard = 32 * (texts.shape[1] + 4) * np.finfo(np.float64).eps * (1 + image_norm * (float(np.sqrt(np.sum(texts * texts, axis=1)).max()) + 1))
    result['full_gallery_score_error_conservative_upper_bound'] = float(image_norm * np.sqrt(np.sum(change * change, axis=1)).max() + guard)
    for direction in ('i2t', 't2i'):
        row = {'winner_changes': int(np.count_nonzero(frozen[f'{direction}_top_indices'] != identity[f'{direction}_top_indices'])),
               'correctness_changes': int(np.count_nonzero(frozen[f'{direction}_correct'] != identity[f'{direction}_correct']))}; errors = []
        for raw in (frozen, identity):
            winner = raw[f'{direction}_top_indices']; query = np.arange(len(winner)); ii, tt = (query, winner) if direction == 'i2t' else (winner, query)
            errors.append(float(np.max(np.abs(np.sum(images[ii] * texts[tt], axis=1, dtype=np.float64) - np.sum(images[ii] * normalized[tt], axis=1, dtype=np.float64)))))
        row['max_absolute_pair_score_error_on_union_of_winners'] = max(errors); result[direction] = row
    return result


def score(args, release, supplement, lock, lock_entry):
    output = root_path(lock['output_root']) / args.encoder / args.dataset
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing to overwrite LABCLIP outcomes')
    core_raw, core_index, core_receipt = release_api.load_prediction_index(release, args.release, supplement, 'core', args.encoder, args.dataset)
    core_lock = read(verify_record(lock['core_lock'])); protocol = read(verify_record(core_lock['protocol']))
    config = {'datasets': {n: {k: e['path'] for k, e in v.items()} for n, v in lock['inputs'][args.encoder].items()}}
    metadata = read(verify_record(protocol['original_training_inputs'][args.encoder]['metadata'])); dataset = core.load_dataset(config, args.dataset, args.encoder, metadata)
    output.mkdir(parents=True, exist_ok=True)
    start = write_json(output / 'prescore_receipt.json', {'release': record(args.release), 'additional_lock': lock_entry, 'core_lock': lock['core_lock'],
          'core_index': core_index, 'core_release_receipt': core_receipt, 'encoder': args.encoder, 'dataset': args.dataset,
          'inputs': lock['inputs'][args.encoder][args.dataset], 'score': SCORE_IDENTITY})
    artifacts = {}; dimension = 512 if args.encoder == 'vit_b32' else 1024
    for seed in (0,) + SEEDS:
        state = lock['states'][f'{args.encoder}__{17 if seed == 0 else seed}']; entry = state['identity_checkpoint' if seed == 0 else 'checkpoint']
        scorer = LABCLIPScorer(load_model(verify_record(entry), dimension, state['native_logit_scale']))
        summary, raw = score_triplets(dataset, scorer) if args.dataset == 'sugarcrepe_pp' else score_retrieval(dataset, scorer); name = 'identity' if seed == 0 else f'labclip_{seed}'
        artifacts[name] = {'state': state, 'checkpoint': entry, 'summary': summary, 'predictions': write_npz(output / f'{name}.npz', raw)}
        if seed == 0: parity = identity_parity(dataset, core_raw['frozen'], raw)
    for e in lock['inputs'][args.encoder][args.dataset].values(): verify_record(e)
    print(json.dumps(write_json(output / 'index.json', {'study': 'sanw_practical_v10_labclip_benchmark_predictions', 'status': 'complete',
          'release': record(args.release), 'additional_lock': lock_entry, 'encoder': args.encoder, 'dataset': args.dataset,
          'start_receipt': start, 'artifacts': artifacts, 'identity_vs_original_frozen': parity})), flush=True)


def load_prediction_index(args, release, supplement, lock, lock_entry, encoder, dataset):
    output = root_path(lock['output_root']) / encoder / dataset; index = read(output / 'index.json'); start = read(verify_record(index['start_receipt']))
    if (index.get('study') != 'sanw_practical_v10_labclip_benchmark_predictions' or index.get('status') != 'complete' or index['release'] != record(args.release)
            or index['additional_lock'] != lock_entry or index['encoder'] != encoder or index['dataset'] != dataset
            or start['release'] != record(args.release) or start['additional_lock'] != lock_entry or start['core_lock'] != lock['core_lock']
            or start['encoder'] != encoder or start['dataset'] != dataset or start['inputs'] != lock['inputs'][encoder][dataset] or start['score'] != SCORE_IDENTITY
            or set(index['artifacts']) != {'identity', *(f'labclip_{s}' for s in SEEDS)}): raise ValueError('LABCLIP prediction identity differs')
    core_raw, core_index, core_receipt = release_api.load_prediction_index(release, args.release, supplement, 'core', encoder, dataset)
    if start['core_index'] != core_index or start['core_release_receipt'] != core_receipt: raise ValueError('LABCLIP frozen reference differs')
    raw = {}
    for name, item in index['artifacts'].items():
        seed = 17 if name == 'identity' else int(name.split('_')[1]); state = lock['states'][f'{encoder}__{seed}']
        if item['state'] != state or item['checkpoint'] != state['identity_checkpoint' if name == 'identity' else 'checkpoint']: raise ValueError('LABCLIP checkpoint differs')
        raw[name] = core.archive(item['predictions'])
    return core_raw, raw, index, record(output / 'index.json'), core_index, core_receipt


def analyze(args, release, supplement, lock, lock_entry):
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing to overwrite LABCLIP effects')
    output.mkdir(parents=True, exist_ok=True); (output / 'paired').mkdir(); effects, draws, paired, indices, descriptive = [], {}, {}, [], []
    for encoder in ENCODERS:
        for dataset in DATASETS:
            core_raw, raw, index, rec, core_index, core_receipt = load_prediction_index(args, release, supplement, lock, lock_entry, encoder, dataset)
            indices.append({'index': rec, 'core_index': core_index, 'core_release_receipt': core_receipt})
            descriptive.append({'encoder': encoder, 'dataset': dataset, 'identity_vs_original_frozen': index['identity_vs_original_frozen'], 'absolute_metrics_by_seed': {n: a['summary'] for n, a in index['artifacts'].items()}})
            for metric in lock['endpoints'][dataset]:
                delta, clusters = paired_seed_differences(dataset, metric, {s: core_raw[f'joint_{s}'] for s in SEEDS}, {s: raw[f'labclip_{s}'] for s in SEEDS})
                effect, samples = bootstrap(delta, clusters); key = f"{encoder}__{dataset}__{metric.replace('.', '_')}__joint_minus_labclip"
                effects.append({**effect, 'effect_id': key, 'encoder': encoder, 'dataset': dataset, 'metric': metric, 'contrast': 'joint_minus_labclip'}); draws[key] = samples
                paired[key] = write_npz(output / 'paired' / f'{key}.npz', {'difference_by_seed': delta, 'cluster_ids': clusters, 'fixed_seeds': np.asarray(SEEDS)})
    if len(effects) != 10: raise ValueError('All ten LABCLIP effects required')
    print(json.dumps(write_json(output / 'analysis.json', {'study': 'sanw_practical_v10_labclip_benchmark_analysis', 'release': record(args.release),
          'additional_lock': lock_entry, 'core_lock': lock['core_lock'], 'indices': indices, 'effects': effects, 'paired_artifacts': paired,
          'bootstrap': write_npz(output / 'bootstrap.npz', draws), 'effect_count': 10, 'descriptive_baseline_and_identity': descriptive,
          'core_practical_result_used_to_filter_effects': False, 'candidate_selection_allowed': False, 'fresh_confirmation_inclusion_allowed': False,
          'new_gate': None, 'evaluation_status': lock['evaluation_status']})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True); q = sub.add_parser('lock')
    for k in ('core-lock', 'core-lock-sha256', 'contract', 'source-audit', 'output-root', 'output'): q.add_argument('--' + k, required=True)
    q.add_argument('--runs', nargs=6, required=True); q.add_argument('--training-audits', nargs=6, required=True); q = sub.add_parser('audit-training')
    for k in ('core-lock', 'core-lock-sha256', 'contract', 'run', 'output'): q.add_argument('--' + k, required=True)
    for mode in ('score', 'analyze'):
        q = sub.add_parser(mode); q.add_argument('--release', required=True); q.add_argument('--release-sha256', required=True)
        if mode == 'score': q.add_argument('--encoder', choices=ENCODERS, required=True); q.add_argument('--dataset', choices=DATASETS, required=True)
        else: q.add_argument('--output', required=True)
    args = p.parse_args(); core.development.require_evaluation_threads()
    if args.mode == 'lock': print(json.dumps(write_json(args.output, build_lock(args.core_lock, args.core_lock_sha256, args.contract, args.source_audit, args.runs, args.output_root, training_audits=args.training_audits))), flush=True)
    elif args.mode == 'audit-training': audit_training(args)
    else: (score if args.mode == 'score' else analyze)(args, *load_released(args))
if __name__ == '__main__': main()
