#!/usr/bin/env python3
"""Post-review descriptive retrieval on the existing e-ViL-held-out Flickr400.

This creates a new protocol and new outputs only. It does not select models,
train adapters, modify the frozen study, or claim the standard Flickr1000 task.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcr.adapters import ResidualAdapter
from gcr.evaluation import evaluate_retrieval, RETRIEVAL_TIE_POLICY
from evaluate_study import adapted_features

OUTPUT = ROOT / 'results/review_followup_flickr400'
METRICS = [f'{direction}.{metric}' for direction in ('i2t', 't2i') for metric in ('r1', 'r5', 'r10')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def metric_value(metrics, key):
    direction, name = key.split('.')
    return metrics[direction][name]


def audit_ranks(images, texts, image_ids, text_ids, pairs, predictions):
    """Independent full sorting for evenly spaced queries in both directions."""
    image_lookup = {name: i for i, name in enumerate(image_ids)}
    text_lookup = {name: i for i, name in enumerate(text_ids)}
    relevant_images = [[] for _ in image_ids]
    relevant_texts = [[] for _ in text_ids]
    for pair in pairs:
        ii, ti = image_lookup[pair['image_id']], text_lookup[pair['text_id']]
        relevant_images[ii].append(ti)
        relevant_texts[ti].append(ii)
    counts = {}
    for direction, queries, candidates, relevant in (
        ('i2t', images, texts, relevant_images), ('t2i', texts, images, relevant_texts)
    ):
        chosen = np.unique(np.linspace(0, len(queries) - 1, 11, dtype=np.int64))
        for index in chosen:
            scores = queries[index].astype(np.float64) @ candidates.astype(np.float64).T
            order = np.lexsort((np.arange(len(candidates)), -scores))
            actual = int(np.flatnonzero(np.isin(order, relevant[index]))[0]) + 1
            if actual != int(predictions[f'{direction}_ranks'][index]):
                raise ValueError(f'Independent ranking mismatch: {direction}, query {index}')
        counts[direction] = len(chosen)
    return counts


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = ROOT / 'data/visual_entailment/manifest.json'
    feature_path = ROOT / 'results/features/visual_entailment/features.npz'
    metadata_path = feature_path.with_name('metadata.json')
    selection_path = ROOT / 'results/study/selection.json'
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection['held_out_data_used'] or selection['selected_checkpoint_count'] != 36:
        raise ValueError('Require the unchanged validation-selected 36-state study')
    if metadata['manifest_sha256'] != sha(manifest_path) or metadata['features_sha256'] != sha(feature_path):
        raise ValueError('Feature/manifest provenance mismatch')
    image_rows = manifest['images']
    text_rows = manifest['texts']
    selected_image_indices = [i for i, row in enumerate(image_rows) if row['split'] == 'test']
    image_ids = [image_rows[i]['id'] for i in selected_image_indices]
    image_set = set(image_ids)
    excluded_image_ids = {row['id'] for row in image_rows if row['split'] != 'test'}
    pairs = [p for p in manifest['pairs'] if p['relation'] == 'source' and p['image_id'] in image_set]
    selected_text_ids = {p['text_id'] for p in pairs}
    selected_text_indices = [i for i, row in enumerate(text_rows) if row['id'] in selected_text_ids]
    text_ids = [text_rows[i]['id'] for i in selected_text_indices]
    excluded_text_ids = {p['text_id'] for p in manifest['pairs'] if p['image_id'] in excluded_image_ids}
    if len(image_ids) != 400 or len(image_set) != 400 or len(text_ids) != 2000 or len(pairs) != 2000:
        raise ValueError('Expected exactly 400 held-out images and 2000 unique source-caption IDs')
    if selected_text_ids & excluded_text_ids or image_set & excluded_image_ids:
        raise ValueError('Split leakage detected')
    if any(sum(p['image_id'] == image_id for p in pairs) != 5 for image_id in image_ids):
        raise ValueError('Every image must have five source captions')
    if any(sum(p['text_id'] == text_id for p in pairs) != 1 for text_id in text_ids):
        raise ValueError('Every caption must have one annotated source image')
    texts_by_id = {row['id']: row['text'] for row in text_rows}
    candidate_strings = {texts_by_id[t] for t in text_ids}
    excluded_source_strings = {texts_by_id[p['text_id']] for p in manifest['pairs']
                               if p['relation'] == 'source' and p['image_id'] in excluded_image_ids}
    string_overlap = len(candidate_strings & excluded_source_strings)
    if string_overlap:
        raise ValueError('Exact source-caption strings shared with non-test splits')
    with np.load(feature_path, allow_pickle=False) as z:
        if z['image_ids'].tolist() != [row['id'] for row in image_rows] or z['text_ids'].tolist() != [row['id'] for row in text_rows]:
            raise ValueError('Feature ID ordering differs from manifest')
        features = {'image_features': z['image_features'][selected_image_indices],
                    'text_features': z['text_features'][selected_text_indices]}
    for name, values in features.items():
        if not np.isfinite(values).all() or values.shape[1] != 512:
            raise ValueError(f'Invalid features: {name}')
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError(f'Nonunit features: {name}')
    runs = [{'run_id': 'frozen', 'method': 'frozen', 'seed': None, 'epoch': 0,
             'checkpoint': None, 'checkpoint_sha256': None}]
    for method, entry in sorted(selection['methods'].items()):
        if sorted(r['seed'] for r in entry['runs']) != [17, 29, 43]:
            raise ValueError('Missing selected seed')
        for run in sorted(entry['runs'], key=lambda row: row['seed']):
            if sha(ROOT / run['checkpoint']) != run['checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            runs.append({'run_id': f'{method}_seed_{run["seed"]}', 'method': method,
                         'seed': run['seed'], 'epoch': run['epoch'], 'learning_rate': entry['learning_rate'],
                         'checkpoint': run['checkpoint'], 'checkpoint_sha256': run['checkpoint_sha256']})
    inputs = {name: {'bytes': (ROOT / name).stat().st_size, 'sha256': sha(ROOT / name)} for name in (
        'data/visual_entailment/manifest.json', 'data/visual_entailment/selection.json',
        'data/visual_entailment/provenance.json', 'results/features/visual_entailment/features.npz',
        'results/features/visual_entailment/metadata.json', 'results/study/selection.json',
        'docs/RERUN_PROTOCOL.json', 'scripts/evaluate_flickr_source_pool.py', 'scripts/evaluate_study.py',
        'src/gcr/adapters.py', 'src/gcr/evaluation.py')}
    protocol = {
        'schema_version': 1, 'created_utc_before_scoring': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_before_scoring', 'evidence_type': 'post_review_descriptive_followup',
        'name': 'e-ViL-held-out Flickr400 source-caption full-pool retrieval',
        'not_claimed': ['standard Karpathy Flickr1000 retrieval', 'matched-epoch comparison',
                        'original preregistered endpoint', 'freshly uninspected held-out dataset'],
        'scope': 'Existing 400 e-ViL test images, five source captions per image; hypotheses excluded.',
        'candidate_pool_images': 400, 'candidate_pool_texts': 2000,
        'image_ids_in_candidate_order': image_ids, 'text_ids_in_candidate_order': text_ids,
        'source_relevance_pairs': pairs, 'text_strings_in_candidate_order': [texts_by_id[t] for t in text_ids],
        'duplicate_candidate_text_strings': len(text_ids) - len(candidate_strings),
        'relevance_rule': 'Only the five annotated source captions per image and their annotated source image count as relevant; identical strings retain distinct IDs.',
        'split_checks': {'test_image_overlap_with_non_test': 0, 'source_text_id_overlap_with_non_test': 0,
                         'exact_source_string_overlap_with_non_test': string_overlap},
        'model_selection': 'All original validation-selected states reused without tuning or reselection; selected epochs may differ.',
        'metrics': METRICS, 'comparison_plan': 'Report each run, three-seed means/sample SD, and per-seed/mean percentage-point differences versus frozen and same-seed source-positive clip.',
        'inference': 'Descriptive only; no confirmatory significance claim or post-hoc winner selection.',
        'score_dtype': 'float64', 'adaptation_dtype': 'float32',
        'frozen_normalization': 'Same final float32 L2 normalization used by evaluate_study.adapted_features.',
        'tie_policy': RETRIEVAL_TIE_POLICY, 'block_size': 128,
        'inputs': inputs, 'selection_ledger_sha256': selection['ledger_sha256'], 'runs': runs,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                        'torch_threads': 2, 'blas_threads': {key: os.getenv(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / 'protocol.json'
    write_new(protocol_path, protocol)
    protocol_hash = sha(protocol_path)
    print(json.dumps({'protocol_frozen': str(protocol_path.relative_to(ROOT)), 'sha256': protocol_hash,
                      'images': len(image_ids), 'source_captions': len(text_ids), 'runs': len(runs)}), flush=True)
    records = []
    for run in runs:
        adapter = None
        if run['checkpoint']:
            payload = torch.load(ROOT / run['checkpoint'], map_location='cpu', weights_only=True)
            for key in ('method', 'seed', 'epoch'):
                if payload[key] != run[key]:
                    raise ValueError(f'Checkpoint metadata mismatch: {run["run_id"]}, {key}')
            if payload['ledger_sha256'] != selection['ledger_sha256']:
                raise ValueError('Checkpoint ledger mismatch')
            adapter = ResidualAdapter(dim=512)
            adapter.load_state_dict(payload['state_dict'], strict=True)
            adapter.eval()
        images, texts = adapted_features(features, adapter)
        metrics, predictions = evaluate_retrieval(images, texts, image_ids, text_ids, pairs, block_size=128)
        audit = audit_ranks(images, texts, image_ids, text_ids, pairs, predictions)
        path = OUTPUT / f'{run["run_id"]}_ranks.npz'
        if path.exists():
            raise FileExistsError(path)
        compact = {key: value for key, value in predictions.items()
                   if key.endswith('_ranks') or key.endswith('_best_relevant_indices') or key.endswith('_best_relevant_scores')}
        np.savez_compressed(path, **compact)
        records.append({**run, 'metrics': metrics, 'protocol_sha256': protocol_hash,
                        'predictions': str(path.relative_to(ROOT)), 'predictions_sha256': sha(path),
                        'query_ids': 'protocol.json candidate orders; i2t follows image IDs, t2i follows text IDs',
                        'independent_full_sort_queries_verified': audit})
        print(json.dumps({'run': run['run_id'], 'i2t_r1': metrics['i2t']['r1'], 't2i_r1': metrics['t2i']['r1']}), flush=True)
    frozen = records[0]['metrics']
    clip_by_seed = {r['seed']: r['metrics'] for r in records if r['method'] == 'clip'}
    for record in records:
        record['differences_vs_frozen_pp'] = {key: 100 * (metric_value(record['metrics'], key) - metric_value(frozen, key)) for key in METRICS}
        if record['seed'] is not None:
            record['differences_vs_same_seed_clip_pp'] = {key: 100 * (metric_value(record['metrics'], key) - metric_value(clip_by_seed[record['seed']], key)) for key in METRICS}
    means = {}
    for method in ['frozen'] + sorted(selection['methods']):
        members = [record for record in records if record['method'] == method]
        means[method] = {'runs': len(members), 'seeds': [r['seed'] for r in members],
                         'selected_epochs': [r['epoch'] for r in members], 'metrics': {}}
        for key in METRICS:
            values = np.asarray([metric_value(r['metrics'], key) for r in members])
            row = {'mean': float(values.mean()), 'sample_std': float(values.std(ddof=1)) if len(values) > 1 else None,
                   'per_seed_values': values.tolist(), 'difference_vs_frozen_pp': float(100 * (values.mean() - metric_value(frozen, key)))}
            if method != 'frozen':
                delta = [r['differences_vs_same_seed_clip_pp'][key] for r in members]
                row['difference_vs_same_seed_clip_pp'] = {'mean': float(np.mean(delta)), 'per_seed': delta}
            means[method]['metrics'][key] = row
    write_new(OUTPUT / 'per_run_results.json', {'protocol_sha256': protocol_hash, 'runs': records})
    summary = {'protocol_sha256': protocol_hash, 'status': 'complete', 'evidence_type': protocol['evidence_type'],
               'candidate_pool': {'images': 400, 'source_captions': 2000}, 'run_count': len(records),
               'limitations': protocol['not_claimed'], 'methods': means, 'elapsed_seconds': time.monotonic() - started}
    write_new(OUTPUT / 'summary.json', summary)
    with (OUTPUT / 'summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'metric', 'mean_percent', 'sample_std_percentage_points', 'difference_vs_frozen_pp', 'difference_vs_source_clip_pp'])
        for method, info in means.items():
            for key, values in info['metrics'].items():
                writer.writerow([method, key, 100 * values['mean'], None if values['sample_std'] is None else 100 * values['sample_std'],
                                 values['difference_vs_frozen_pp'], values.get('difference_vs_same_seed_clip_pp', {}).get('mean')])
    checks = {'status': 'passed', 'protocol_sha256': protocol_hash, 'run_count': len(records),
              'original_inputs_unchanged': all(sha(ROOT / name) == value['sha256'] for name, value in inputs.items()),
              'checkpoint_hashes_unchanged': all(not r['checkpoint'] or sha(ROOT / r['checkpoint']) == r['checkpoint_sha256'] for r in runs),
              'independent_sorted_queries': sum(sum(r['independent_full_sort_queries_verified'].values()) for r in records),
              'protocol_unchanged': sha(protocol_path) == protocol_hash,
              'ranking_files_valid': all(sha(ROOT / r['predictions']) == r['predictions_sha256'] for r in records)}
    if not all(checks[key] for key in ('original_inputs_unchanged', 'checkpoint_hashes_unchanged', 'protocol_unchanged', 'ranking_files_valid')):
        raise ValueError('Final provenance verification failed')
    write_new(OUTPUT / 'execution_audit.json', checks)
    print(json.dumps({'status': 'complete', 'seconds': summary['elapsed_seconds'], 'audit': checks}), flush=True)


if __name__ == '__main__':
    main()
