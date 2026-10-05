#!/usr/bin/env python3
"""Sequence the unchanged certificate checker as each complete index arrives."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from audit_allocation_distillation_independent import Audit, digest, read_json
from audit_retention_certificates_independent import audit_index, PROTOCOL_SHA256

SOURCE_NAMES = ('scripts/audit_retention_certificates_independent.py',
                'scripts/audit_allocation_distillation_independent.py',
                'recovery/current_turn_audit/run_certificate_audit_20261004.py')
source_hashes = {name: digest(ROOT / name) for name in SOURCE_NAMES}
protocol = 'results/strengthen_retention/protocol_v3.json'
lock_path = 'results/strengthen_retention/selection_lock_recovered.json'
lock_hash = 'ec2e816e74af2cc93577baec6dd0889e6ed4bb44402591ada25d2895be0b926a'
indices = {encoder: f'results/strengthen_retention/{encoder}/diagnostics/index.json'
           for encoder in ('vit_b32', 'rn50')}
configs = {encoder: f'recovery/current_input_recovery/datasets_{encoder}.json' for encoder in indices}
audit, maxima = Audit(ROOT), defaultdict(float)
audit.check_hash(protocol, PROTOCOL_SHA256)
audit.check_hash(lock_path, lock_hash)
lock = read_json(ROOT / lock_path)
if lock['protocol_sha256'] != PROTOCOL_SHA256 or set(lock['encoders']) != set(indices):
    raise ValueError('Both original-retention selections must remain locked')
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
results, locked_inputs = {}, {protocol: PROTOCOL_SHA256, lock_path: lock_hash}
for encoder, index_path in indices.items():
    started = time.monotonic()
    while not (ROOT / index_path).exists() or read_json(ROOT / index_path).get('status') != 'complete':
        if time.monotonic() - started > 1800:
            raise TimeoutError(f'Complete certificate index did not arrive: {encoder}')
        print(json.dumps({'waiting_for_complete_diagnostic_index': encoder}), flush=True)
        time.sleep(10)
    for path, expected in source_hashes.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Independent audit source changed during execution')
    index_hash, config_hash = digest(ROOT / index_path), digest(ROOT / configs[encoder])
    if encoder == 'vit_b32' and index_hash != '57d6a5efaa9b9c31d72ba1f92257a3ecc9b549ee37f4f972e0b444a03ad88377':
        raise ValueError('Completed ViT diagnostic index changed')
    identity = {'encoder': encoder, 'protocol_sha256': PROTOCOL_SHA256, 'selection_lock_sha256': lock_hash,
                'index_sha256': index_hash, 'dataset_config_sha256': config_hash, 'source_sha256': source_hashes}
    cache_path = ROOT / f'recovery/current_turn_audit/retention_certificate_{encoder}_partial_audit.json'
    if cache_path.exists():
        cache = read_json(cache_path)
        if cache['status'] != 'encoder_checks_passed' or cache['identity'] != identity:
            raise ValueError('Existing per-encoder audit checkpoint differs')
        for path, expected in cache['verified_input_sha256'].items():
            actual = digest(ROOT / path)
            if actual != expected:
                raise ValueError(f'An independently verified input changed: {path}')
            audit.hash_cache[(ROOT / path).resolve()] = actual
        results[encoder] = cache['result']
        audit.counts.update(cache['audit_counts'])
        for key, value in cache['max_absolute_differences'].items():
            maxima[key] = max(maxima[key], value)
        print(json.dumps({'reused_verified_encoder_audit': encoder}), flush=True)
    else:
        before = Counter(audit.counts)
        actual_encoder, result = audit_index(audit, index_path, configs[encoder], lock, lock_hash, 128, maxima)
        if actual_encoder != encoder:
            raise ValueError('Certificate index encoder differs')
        results[encoder] = result
        cache = {'schema_version': 1, 'status': 'encoder_checks_passed', 'identity': identity, 'result': result,
                 'audit_counts': dict(Counter(audit.counts) - before), 'max_absolute_differences': dict(maxima),
                 'verified_input_sha256': {str(path.relative_to(ROOT)): value for path, value in audit.hash_cache.items()}}
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'independent_encoder_certificate_audit': encoder, 'status': 'passed'}), flush=True)
    if digest(ROOT / index_path) != index_hash or digest(ROOT / configs[encoder]) != config_hash:
        raise ValueError('Certificate index or dataset config changed during audit')
    locked_inputs.update({index_path: index_hash, configs[encoder]: config_hash})
for path, expected in {**source_hashes, **locked_inputs}.items():
    if digest(ROOT / path) != expected:
        raise ValueError('A combined audit input changed')
result = {'schema_version': 1, 'status': 'passed', 'protocol_sha256': PROTOCOL_SHA256,
          'input_sha256': locked_inputs, 'source_sha256': source_hashes,
          'independent_implementation': 'Unchanged audit_index checker recomputes all full-gallery scores from cached features and weights, then enumerates feasible positive-set pooling sizes. No production adapter, rank, or certificate implementation is imported. Encoder checks run sequentially after their complete indices arrive.',
          'scope': 'Same complete galleries only; forward teacher-to-student KL at T=2, without T-squared scaling.',
          'interpretation': 'Counts are state-query observations over reused galleries. Nonzero-update coverage is separate from unchanged-state coverage. Coverage does not establish caption improvement or out-of-gallery guarantees.',
          'floating_point_caveat': 'Independent floating-point recomputation, not an interval-arithmetic proof.',
          'false_certificates': 0, 'encoders': results, 'max_absolute_differences': dict(maxima),
          'checks': dict(audit.counts), 'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
          'torch': str(torch.__version__), 'torch_threads': 2, 'block_size': 128}}
output = ROOT / 'recovery/current_turn_audit/retention_certificates_independent_audit.json'
if output.exists() and read_json(output) != result:
    raise ValueError('Existing combined certificate audit differs')
output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': 'passed', 'encoders': list(results), 'false_certificates': 0,
                  'receipt_sha256': digest(output)}), flush=True)
