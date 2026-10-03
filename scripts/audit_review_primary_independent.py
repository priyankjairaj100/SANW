#!/usr/bin/env python3
"""Independent integer-count audit of follow-up primary raw-rank effects.

This audit imports no project analysis/evaluation code. It reconstructs effects
from manifest ownership and integer retrieval successes, and bootstraps integer
image-cluster totals before a single rational division.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-inventory', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = ROOT / 'results/review_followup/analysis'
    protocol_path = ROOT / 'docs/REVIEW_FOLLOWUP_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    assert digest(protocol_path) == '3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34'
    states_file = ROOT / 'results/review_followup/state_manifest.json'
    state_manifest = json.loads(states_file.read_text())
    terminal_index_file = ROOT / 'results/review_followup/evaluation/terminal_index.json'
    terminal_index = json.loads(terminal_index_file.read_text())
    index = {row['state_id']: row for row in terminal_index['runs']}
    primary_file = output / 'primary_contrasts.json'
    primary = json.loads(primary_file.read_text())
    saved_samples_file = output / 'primary_bootstrap_samples.npz'
    saved_samples = np.load(saved_samples_file, allow_pickle=False)
    dataset_file = ROOT / 'data/review_followup/e_vil_test1000/manifest.json'
    dataset = json.loads(dataset_file.read_text())
    image_ids = [str(row['id']) for row in dataset['images']]
    text_ids = [str(row['id']) for row in dataset['texts']]
    assert len(image_ids) == len(set(image_ids)) == 1000
    assert len(text_ids) == len(set(text_ids)) == 5000
    owner = {}
    for pair in dataset['pairs']:
        assert pair['relation'] == 'source' and pair['text_id'] not in owner
        owner[pair['text_id']] = pair['image_id']
    assert Counter(owner.values()) == Counter({image: 5 for image in image_ids})
    assert set(owner) == set(text_ids)
    owners = [owner[text] for text in text_ids]
    cluster_ids = sorted(image_ids)
    image_position = {image: i for i, image in enumerate(image_ids)}
    text_cluster_positions = [[i for i, image in enumerate(owners) if image == cluster] for cluster in cluster_ids]
    assert all(len(indices) == 5 for indices in text_cluster_positions)
    successes, raw_bindings = {}, []
    seeds = (17, 29, 43)
    rates = (.0001, .0003, .001)
    methods = ('source', 'supported') + tuple(f'{family}_draw_{draw}' for family in ('count_only', 'score_stratified') for draw in range(3))
    terminal = [s for s in state_manifest['states'] if s['epoch'] == 10]
    assert {(s['method'], s['learning_rate'], s['seed']) for s in terminal} == {(m, r, seed) for m in methods for r in rates for seed in seeds}
    maximum_raw_metric_difference = 0.
    for state in terminal:
        reference = index[state['state_id']]['datasets']['e_vil_test1000']
        archive = ROOT / reference['predictions']
        archive_sha = digest(archive)
        assert archive_sha == reference['predictions_sha256']
        raw_bindings.append({'state_id': state['state_id'], 'sha256': archive_sha})
        with np.load(archive, allow_pickle=False) as raw:
            assert raw['image_ids'].tolist() == image_ids
            assert raw['text_ids'].tolist() == text_ids
            assert raw['text_source_image_ids'].tolist() == owners
            for direction, count, maximum_rank in [('i2t', 1000, 5000), ('t2i', 5000, 1000)]:
                ranks = raw[f'{direction}_ranks']
                assert ranks.shape == (count,) and np.issubdtype(ranks.dtype, np.integer)
                assert np.all((ranks >= 1) & (ranks <= maximum_rank))
                values = np.asarray(ranks == 1, dtype=np.int64)
                successes[(state['method'], state['learning_rate'], state['seed'], direction)] = values
                error = abs(float(values.sum()) / count - reference['metrics'][direction]['r1'])
                assert error < 1e-14
                maximum_raw_metric_difference = max(maximum_raw_metric_difference, error)
    assert len(primary['contrasts']) == 18 and len(saved_samples.files) == 18
    expected_effects = {(rate, comparator, metric) for rate in rates for comparator in ('source', 'count_only', 'score_stratified') for metric in ('i2t.r1', 't2i.r1')}
    assert {(row['learning_rate'], row['right'], row['metric']) for row in primary['contrasts']} == expected_effects
    results = []
    for effect in primary['contrasts']:
        lr, comparator, direction = effect['learning_rate'], effect['right'], effect['metric'][:3]
        comparator_methods = [comparator] if comparator == 'source' else [f'{comparator}_draw_{i}' for i in range(3)]
        assert effect['comparator_draw_methods'] == comparator_methods and effect['seed_ids'] == list(seeds)
        left = np.stack([successes[('supported', lr, seed, direction)] for seed in seeds])
        right = np.stack([[successes[(method, lr, seed, direction)] for seed in seeds] for method in comparator_methods])
        draw_count, _, query_count = right.shape
        # Exact integer numerator preserves draw-within-seed weights without
        # float intermediate averaging or duplicating supported fits.
        per_seed_numerator = draw_count * left - right.sum(axis=0)
        query_numerator = per_seed_numerator.sum(axis=0)
        denominator = len(seeds) * draw_count
        difference = int(query_numerator.sum()) / (denominator * query_count)
        seed_differences = per_seed_numerator.sum(axis=1) / (draw_count * query_count)
        supported_seed_values = left.sum(axis=1) / query_count
        comparator_draw_by_seed_values = right.sum(axis=2) / query_count
        if direction == 'i2t':
            cluster_numerators = query_numerator[[image_position[image] for image in cluster_ids]]
            captions_per_cluster = 1
        else:
            cluster_numerators = np.array([int(query_numerator[indices].sum()) for indices in text_cluster_positions], dtype=np.int64)
            captions_per_cluster = 5
        assert len(cluster_numerators) == 1000
        samples = np.empty(10000, dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(20261004))
        # Deliberately different batch size from the production implementation.
        for start in range(0, 10000, 137):
            stop = min(start + 137, 10000)
            selected_clusters = rng.integers(1000, size=(stop - start, 1000), dtype=np.int64)
            samples[start:stop] = cluster_numerators[selected_clusters].sum(axis=1) / (1000 * denominator * captions_per_cluster)
        tail = .05 / (2 * 18)
        interval = np.quantile(samples, [tail, 1-tail], method='linear')
        expected_samples = saved_samples[effect['effect_id']]
        errors = {
            'mean': abs(difference - effect['difference']),
            'seed_differences': float(np.max(np.abs(seed_differences - effect['seed_differences']))),
            'supported_seed_values': float(np.max(np.abs(supported_seed_values - effect['supported_seed_values']))),
            'comparator_draw_by_seed_values': float(np.max(np.abs(comparator_draw_by_seed_values - effect['comparator_draw_by_seed_values']))),
            'bootstrap_samples': float(np.max(np.abs(samples - expected_samples))),
            'confidence_quantiles': float(np.max(np.abs(interval - [effect['ci_lower'], effect['ci_upper']]))),
            'percentage_points': abs(100 * difference - effect['difference_percentage_points']),
        }
        assert max(errors.values()) < 1e-12, (effect['effect_id'], errors)
        assert effect['images'] == 1000 and effect['items'] == query_count and effect['training_seeds'] == 3
        assert effect['family_size'] == 18 and effect['replicates'] == 10000 and effect['bootstrap_seed'] == 20261004
        assert effect['confidence'] == 1 - .05 / 18 and effect['quantile_method'] == 'linear'
        results.append({'effect_id': effect['effect_id'], 'learning_rate': lr, 'metric': effect['metric'],
                        'comparator': comparator, 'draws_averaged_within_seed': draw_count,
                        'training_seeds': list(seeds), 'image_clusters': 1000, 'queries_per_cluster': captions_per_cluster,
                        'difference': difference, 'ci_lower': float(interval[0]), 'ci_upper': float(interval[1]),
                        'interval_excludes_zero': bool(interval[0] > 0 or interval[1] < 0),
                        'max_absolute_differences': errors})
    inventory_result = {'status': 'not_requested'}
    if args.baseline_inventory:
        inventory = json.loads(args.baseline_inventory.read_text())
        unchanged, missing, changed = [], [], []
        for entry in inventory['files']:
            path = ROOT / entry['path']
            if not path.exists():
                missing.append(entry['path'])
            elif digest(path) != entry['sha256']:
                changed.append(entry['path'])
            else:
                unchanged.append(entry['path'])
        assert len(inventory['files']) == 732 and len(unchanged) == 729 and len(missing) == 3 and not changed, (len(unchanged), missing, changed)
        inventory_result = {'status': 'passed', 'inventory_sha256': digest(args.baseline_inventory),
                            'inventory_file_count': 732, 'scientific_files_byte_unchanged': 729,
                            'reference_snapshot_commit': 'a04cd20194ea7d376da8ff2fecc1667ffd05ec54',
                            'packaging_only_paths_absent_from_project': missing, 'changed': changed,
                            'verified_scientific_paths': unchanged}
    report = {'status': 'passed', 'independent_implementation': 'No project imports. Integer success counts and integer cluster numerators; separate PCG64 sampling with batch size137.',
              'protocol_sha256': digest(protocol_path), 'state_manifest_sha256': digest(states_file),
              'terminal_index_sha256': digest(terminal_index_file), 'primary_contrasts_sha256': digest(primary_file),
              'saved_bootstrap_samples_sha256': digest(saved_samples_file), 'test_manifest_sha256': digest(dataset_file),
              'audit_source_sha256': digest(Path(__file__)), 'epoch': 10, 'terminal_states': 72, 'rank_vectors': 144,
              'effects': 18, 'bootstrap_replicates_per_effect': 10000, 'bootstrap_values_compared': 180000,
              'maximum_raw_metric_difference': maximum_raw_metric_difference,
              'maximum_primary_comparison_difference': max(max(row['max_absolute_differences'].values()) for row in results),
              'confidence': 1 - .05/18, 'quantile_probabilities': [.05/36, 1-.05/36],
              'cluster_ownership': {'images': 1000, 'source_captions': 5000, 'unique_captions_per_image': 5, 'all72archives_match_manifest': True},
              'contrasts': results, 'raw_prediction_bindings': raw_bindings, 'baseline_preservation': inventory_result,
              'seconds': time.monotonic()-started}
    path = output/'independent_primary_audit.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: report[key] for key in ['status','effects','bootstrap_values_compared','maximum_primary_comparison_difference','seconds']}))

if __name__ == '__main__':
    main()
