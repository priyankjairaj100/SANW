"""Frozen nonlinear-adapter and independent-encoder replication programs.

Fits and selection read development data only. The original implementation is
imported without alteration; all new outputs carry their own immutable ledger.
Every epoch is hashed and audited; only two selector winners and epoch 10 are
retained. Held-out scoring consumes the exported evaluation manifest.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adapters import ResidualAdapter
from .losses import contrastive_loss
from .review_controls import apply_promotion_assignment, load_assignment, make_promotion_assignment, save_assignment
from .review_training import (SourceRetrievalPool, _atomic_npz, candidate_path, ensure_review_ledger,
                              file_record, policy_parts, state_id, validate_source_retrieval)
from .training import (FeatureDataset, StudyConfig, atomic_json, canonical_json, file_lock,
                       seed_everything, sha256_file, validate_adapter)

POLICIES = ('source', 'supported', 'score_stratified_draw_0',
            'score_stratified_draw_1', 'score_stratified_draw_2')
SELECTORS = ('native', 'source_retrieval')
PROTOCOLS = {'nonlinear': 'docs/STRENGTHEN_REPLICATION_PROTOCOL.json',
             'rn50': 'docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json'}
OUTPUTS = {'nonlinear': 'results/strengthen_replication', 'rn50': 'results/strengthen_second_encoder'}
RN50_WEIGHTS_SHA256 = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'


class NonlinearResidualAdapter(nn.Module):
    """Independent bias-free dim->128->dim ReLU residual branches."""

    def __init__(self, dim: int = 512, bottleneck: int = 128):
        super().__init__()
        if dim <= 0 or bottleneck <= 0:
            raise ValueError('Adapter dimensions must be positive.')
        self.image = nn.Sequential(nn.Linear(dim, bottleneck, bias=False), nn.ReLU(),
                                   nn.Linear(bottleneck, dim, bias=False))
        self.text = nn.Sequential(nn.Linear(dim, bottleneck, bias=False), nn.ReLU(),
                                  nn.Linear(bottleneck, dim, bias=False))
        nn.init.zeros_(self.image[2].weight)
        nn.init.zeros_(self.text[2].weight)

    def encode_image(self, values):
        return F.normalize(values + self.image(values), dim=-1)

    def encode_text(self, values):
        return F.normalize(values + self.text(values), dim=-1)

    def forward(self, images, texts):
        return self.encode_image(images), self.encode_text(texts)


def make_adapter(kind: str, dimension: int, bottleneck: int = 128):
    if kind == 'nonlinear':
        return NonlinearResidualAdapter(dimension, bottleneck)
    if kind == 'linear':
        return ResidualAdapter(dimension)
    raise ValueError(f'Unknown adapter kind: {kind}')


def load_replication_adapter(checkpoint: dict[str, Any]):
    adapter = make_adapter(**checkpoint['adapter'])
    adapter.load_state_dict(checkpoint['state_dict'], strict=True)
    return adapter.eval()


def adapter_spec(setting: str, dimension: int | None = None) -> dict[str, Any]:
    if setting not in PROTOCOLS:
        raise ValueError(f'Unknown replication setting: {setting}')
    return {'kind': 'nonlinear' if setting == 'nonlinear' else 'linear',
            'dimension': dimension if dimension is not None else (512 if setting == 'nonlinear' else 1024),
            'bottleneck': 128}


def _original_config(repository: Path):
    original = json.loads((repository / 'results/study/ledger.json').read_text())
    identity = original['identity']
    if hashlib.sha256(canonical_json(identity)).hexdigest() != original['ledger_sha256']:
        raise ValueError('Original study ledger identity is invalid.')
    for name, digest in identity['source_sha256'].items():
        if sha256_file(repository / name) != digest:
            raise ValueError(f'Original frozen source changed: {name}')
    if sha256_file(repository / identity['protocol']['path']) != identity['protocol']['sha256']:
        raise ValueError('Original frozen protocol changed.')
    hp = dict(identity['hyperparameters'])
    for name in ('methods', 'learning_rates', 'seeds', 'adam_betas', 'validation_positives'):
        hp[name] = tuple(hp[name])
    config = StudyConfig(**hp)
    config.validate()
    return original, config


def verify_replication_protocol(path: Path, expected_sha256: str, setting: str,
                                config: StudyConfig, original: dict) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError('Replication protocol SHA256 mismatch; fitting is prohibited.')
    protocol = json.loads(path.read_text())
    checks = {('training', 'policies'): list(POLICIES),
              ('training', 'learning_rates'): [1e-4, 3e-4, 1e-3],
              ('training', 'seeds'): [17, 29, 43], ('training', 'epochs'): 10,
              ('training', 'primary_epoch'): 10,
              ('selection', 'criteria'): list(SELECTORS),
              ('primary_analysis', 'family_size'): 12,
              ('primary_analysis', 'bootstrap_replicates'): 10000,
              ('primary_analysis', 'bootstrap_seed'): 20261013 if setting == 'nonlinear' else 20261014}
    for (section, field), expected in checks.items():
        if protocol.get(section, {}).get(field) != expected:
            raise ValueError(f'Replication protocol disagrees: {section}.{field}')
    expected_hp = dict(original['identity']['hyperparameters'])
    expected_hp['feature_dim'] = 512 if setting == 'nonlinear' else 1024
    if json.loads(canonical_json(dataclasses.asdict(config))) != expected_hp:
        raise ValueError('Replication must inherit every original hyperparameter except RN50 dimension.')
    if setting == 'nonlinear':
        if protocol['training'].get('bottleneck') != 128 or protocol['training'].get('activation') != 'ReLU':
            raise ValueError('Frozen nonlinear architecture is dim->128->dim with ReLU.')
    elif setting == 'rn50':
        if protocol.get('encoder', {}).get('weights_sha256') != RN50_WEIGHTS_SHA256 or protocol['encoder'].get('dimension') != 1024:
            raise ValueError('Frozen second encoder must be official OpenAI RN50.')
    else:
        raise ValueError('Unknown replication setting.')
    return protocol


def load_replication_context(repository: Path, setting: str, features_override: Path | None = None,
                             metadata_override: Path | None = None):
    """Load unchanged manifests and the setting-specific normalized feature cache."""
    original, base = _original_config(repository)
    config = dataclasses.replace(base, feature_dim=512 if setting == 'nonlinear' else 1024)
    inputs = original['identity']['inputs']
    manifest = repository / inputs['manifest']['path']
    if sha256_file(manifest) != inputs['manifest']['sha256']:
        raise ValueError('Original training manifest changed.')
    if setting == 'nonlinear':
        features, metadata = (repository / inputs[k]['path'] for k in ('features', 'metadata'))
        if features_override is None and metadata_override is None:
            for key, path in (('features', features), ('metadata', metadata)):
                if sha256_file(path) != inputs[key]['sha256']:
                    raise ValueError(f'Original {key} digest changed; explicitly bind a recovered cache in a new execution.')
    elif setting == 'rn50':
        features = repository / 'results/strengthen_second_encoder/features/visual_entailment/features.npz'
        metadata = features.with_name('metadata.json')
    else:
        raise ValueError('Unknown replication setting.')
    if (features_override is None) != (metadata_override is None):
        raise ValueError('Recovered features and metadata must be supplied together.')
    features, metadata = features_override or features, metadata_override or metadata
    data = FeatureDataset(repository, manifest, features, metadata, config.feature_dim)
    if features_override is not None:
        if data.metadata.get('features_sha256') != sha256_file(features):
            raise ValueError('Recovered feature metadata must bind the exact feature cache.')
        original_metadata = original['identity']['inputs']['feature_metadata']
        expected_weights = RN50_WEIGHTS_SHA256 if setting == 'rn50' else original_metadata['weights_sha256']
        if data.metadata.get('weights_sha256') != expected_weights:
            raise ValueError('Recovered features use different pretrained encoder weights.')
        if setting == 'nonlinear' and data.logit_scale != float(original_metadata['logit_scale']):
            raise ValueError('Recovered ViT frozen logit scale differs.')
    if setting == 'rn50':
        if data.metadata.get('weights_sha256') != RN50_WEIGHTS_SHA256:
            raise ValueError('Training cache is not the frozen official RN50 encoder.')
        if data.metadata.get('features_sha256') != sha256_file(features):
            raise ValueError('RN50 training feature metadata must bind its exact cache digest.')
    if len(data.split_indices['train']) != 1200 or len(data.split_indices['validation']) != 100:
        raise ValueError('Replication must reuse original 1200 train and 100 native development images.')
    return original, config, data



def audit_assignment_semantics(original: dict, regenerated: dict) -> dict:
    """Audit reconstructed cosines without substituting freshly drawn controls.

    Exact equality is required for every ordered stratum, quota, source and
    selected target identity. Floating-point cosine deviations are reported;
    numerical non-identity never licenses changing an original assignment.
    """
    global_fields = ('assignment_seed', 'control', 'global_image_count', 'global_text_count',
                     'training_image_indices', 'image_iteration_order', 'sampling_rule', 'stratification')
    record_fields = ('image_id', 'image_index', 'source_text_ids', 'source_text_indices',
                     'supported_text_ids', 'supported_text_indices', 'promoted_text_ids',
                     'promoted_text_indices', 'source_count', 'hypothesis_count', 'promotion_count',
                     'score_bins', 'sampling_units', 'source_target_share')
    mismatches = [f'global.{key}' for key in global_fields if original.get(key) != regenerated.get(key)]
    previous, current = original['records'], regenerated['records']
    if len(previous) != len(current):
        mismatches.append('record_count')
    max_cosine_error, cosine_count = 0., 0
    for before, after in zip(previous, current):
        for key in record_fields:
            if before.get(key) != after.get(key):
                mismatches.append(f"{before['image_id']}.{key}")
        if len(before['hypotheses']) != len(after['hypotheses']):
            mismatches.append(f"{before['image_id']}.hypothesis_count")
        for old, new in zip(before['hypotheses'], after['hypotheses']):
            if any(old[k] != new[k] for k in ('text_id', 'text_index', 'reference_relation', 'score_bin')):
                mismatches.append(f"{before['image_id']}.hypothesis_identity")
            error = abs(old['frozen_cosine'] - new['frozen_cosine'])
            if not np.isfinite(error):
                mismatches.append(f"{before['image_id']}.nonfinite_cosine")
            else:
                max_cosine_error = max(max_cosine_error, error)
            cosine_count += 1
    return {'passed': not mismatches, 'training_images_checked': len(previous),
            'hypothesis_cosines_checked': cosine_count, 'max_absolute_frozen_cosine_error': max_cosine_error,
            'ordered_score_strata_and_target_identities_exact': not mismatches,
            'mismatch_count': len(mismatches), 'first_mismatches': mismatches[:30]}


def audit_recovered_control_parity(repository: Path, output: Path, data: FeatureDataset,
                                  protocol_sha256: str) -> dict:
    """Freeze an input/source-bound three-draw recovery audit, aborting on changes."""
    original_ledger = json.loads((repository / 'results/study/ledger.json').read_text())
    original_receipt = json.loads((repository / 'results/review_followup/assignment_files.json').read_text())
    actual_features = sha256_file(data.features_path)
    original_features = original_ledger['identity']['inputs']['features']['sha256']
    provenance = {'manifest_sha256': sha256_file(data.manifest_path),
                  'features_sha256': actual_features, 'protocol_sha256': protocol_sha256}
    reports, regenerated_records = {}, {}
    for policy in POLICIES[2:]:
        item = original_receipt['assignments'][policy]
        original = load_assignment(repository / item['path'], expected_file_sha256=item['sha256'],
            expected_provenance={**provenance, 'features_sha256': original_features,
                                 'protocol_sha256': original_receipt['protocol_sha256']},
            expected_control='score_stratified', expected_seed=policy_parts(policy)[2])
        regenerated = make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
            data.pairs, data.split_indices['train'], 'score_stratified', policy_parts(policy)[2], provenance,
            [row['text'] for row in data.manifest['texts']])
        regenerated_records[policy] = regenerated
        reports[policy] = {**audit_assignment_semantics(original, regenerated),
                           'original_assignment_file': item,
                           'regenerated_content_sha256': regenerated['assignment_content_sha256']}
    regenerated_files = {}
    if all(r['passed'] for r in reports.values()):
        for policy, record in regenerated_records.items():
            regenerated_path = output / 'recovered_assignment_controls' / f'{policy}.json'
            save_assignment(regenerated_path, record)
            regenerated_files[policy] = file_record(repository, regenerated_path)
    receipt = {'schema_version': 1, 'kind': 'recovered_feature_control_semantic_parity',
               'protocol_sha256': protocol_sha256, 'actual_features_sha256': actual_features,
               'original_features_sha256': original_features, 'manifest_sha256': provenance['manifest_sha256'],
               'source_sha256': {name: sha256_file(repository / name) for name in
                                 ('src/gcr/strengthen_replication.py', 'src/gcr/review_controls.py')},
               'passed': all(r['passed'] for r in reports.values()), 'draws': reports,
               'original_assignments_overwritten': False,
               'regenerated_assignments': regenerated_files,
               'regenerated_assignment_directory': str((output / 'recovered_assignment_controls').relative_to(repository)),
               'nonlinear_training_assignment_rule': 'use_original_files_verbatim_after_this_audit'}
    path = output / 'assignment_parity.json'
    with file_lock(output / '.assignment-parity.lock'):
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError('Assignment recovery parity receipt is immutable; use a new output directory.')
        if not path.exists():
            atomic_json(path, receipt)
    if not receipt['passed']:
        raise ValueError('Recovered features change original score strata or promotion identities; fitting prohibited. See assignment_parity.json.')
    return receipt


def prepare_assignments(repository: Path, output: Path, setting: str, protocol_sha256: str,
                        data: FeatureDataset) -> dict:
    records = {}
    parity = None
    if setting == 'nonlinear':
        parity = audit_recovered_control_parity(repository, output, data, protocol_sha256)
    provenance = {'manifest_sha256': sha256_file(data.manifest_path),
                  'features_sha256': sha256_file(data.features_path), 'protocol_sha256': protocol_sha256}
    original_receipt = None
    if setting == 'nonlinear':
        original_receipt = json.loads((repository / 'results/review_followup/assignment_files.json').read_text())
        original_ledger = json.loads((repository / 'results/study/ledger.json').read_text())
        provenance['features_sha256'] = original_ledger['identity']['inputs']['features']['sha256']
    for policy in POLICIES[2:]:
        _, _, seed = policy_parts(policy)
        if setting == 'nonlinear':
            item = original_receipt['assignments'][policy]
            path = repository / item['path']
            # Reuse bytes, original protocol provenance, and original assignment RNG verbatim.
            record = load_assignment(path, expected_file_sha256=item['sha256'],
                                     expected_provenance={**provenance, 'protocol_sha256': original_receipt['protocol_sha256']},
                                     expected_control='score_stratified', expected_seed=seed)
        elif setting == 'rn50':
            path = output / 'assignments' / f'{policy}.json'
            record = make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
                data.pairs, data.split_indices['train'], 'score_stratified', seed, provenance,
                [row['text'] for row in data.manifest['texts']])
            save_assignment(path, record)
        else:
            raise ValueError('Unknown replication setting.')
        if record['source_sha256'] != sha256_file(repository / 'src/gcr/review_controls.py'):
            raise ValueError('Assignment control implementation changed.')
        if record['training_image_indices'] != sorted(data.split_indices['train']):
            raise ValueError('Assignment training image set changed.')
        records[policy] = file_record(repository, path)
    receipt = {'schema_version': 1, 'setting': setting, 'protocol_sha256': protocol_sha256,
               'mode': 'reuse_original_verbatim' if setting == 'nonlinear' else 'recomputed_RN50_frozen_cosines',
               'assignments': records,
               'semantic_parity_audit': file_record(repository, output / 'assignment_parity.json') if parity is not None else None}
    path = output / 'assignment_files.json'
    with file_lock(output / '.assignments.lock'):
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError('Replication assignment receipt is immutable.')
        if not path.exists():
            atomic_json(path, receipt)
    return records


def build_replication_ledger(repository: Path, output: Path, setting: str, protocol_path: Path,
                             protocol_sha256: str, original: dict, config: StudyConfig,
                             data: FeatureDataset, pool: SourceRetrievalPool):
    protocol = verify_replication_protocol(protocol_path, protocol_sha256, setting, config, original)
    receipt = json.loads((output / 'assignment_files.json').read_text())
    if receipt['setting'] != setting or receipt['protocol_sha256'] != protocol_sha256 or set(receipt['assignments']) != set(POLICIES[2:]):
        raise ValueError('All three assignments must be frozen before fitting.')
    parity_receipt = None
    assignment_protocol = protocol_sha256
    assignment_features_sha256 = sha256_file(data.features_path)
    if setting == 'nonlinear':
        parity_record = receipt.get('semantic_parity_audit')
        if not parity_record or sha256_file(repository / parity_record['path']) != parity_record['sha256']:
            raise ValueError('Nonlinear input recovery requires its immutable semantic parity audit.')
        parity_receipt = json.loads((repository / parity_record['path']).read_text())
        if not parity_receipt.get('passed') or parity_receipt.get('actual_features_sha256') != sha256_file(data.features_path):
            raise ValueError('Assignment semantic parity audit failed or used different features.')
        if parity_receipt.get('protocol_sha256') != protocol_sha256 or parity_receipt.get('manifest_sha256') != sha256_file(data.manifest_path):
            raise ValueError('Assignment semantic parity audit used a different protocol or manifest.')
        for name, digest in parity_receipt['source_sha256'].items():
            if sha256_file(repository / name) != digest:
                raise ValueError('Assignment semantic parity implementation changed after audit.')
        original_receipt = json.loads((repository / 'results/review_followup/assignment_files.json').read_text())
        assignment_protocol = original_receipt['protocol_sha256']
        assignment_features_sha256 = original['identity']['inputs']['features']['sha256']
        if receipt['assignments'] != {p: original_receipt['assignments'][p] for p in POLICIES[2:]}:
            raise ValueError('Nonlinear assignments must reuse original frozen files verbatim.')
    assignments = {}
    for policy, item in receipt['assignments'].items():
        assignments[policy] = load_assignment(repository / item['path'], expected_file_sha256=item['sha256'],
            expected_provenance={'manifest_sha256': sha256_file(data.manifest_path),
                'features_sha256': assignment_features_sha256, 'protocol_sha256': assignment_protocol},
            expected_control='score_stratified', expected_seed=policy_parts(policy)[2])
    code = ['src/gcr/strengthen_replication.py', 'scripts/run_strengthen_replication.py',
            'src/gcr/review_training.py', 'src/gcr/review_controls.py', 'src/gcr/training.py',
            'src/gcr/adapters.py', 'src/gcr/losses.py']
    identity = {'schema_version': 1, 'kind': 'post_review_replication', 'setting': setting,
                'test_outcomes_used_for_selection': False, 'protocol': file_record(repository, protocol_path),
                'protocol_contents': protocol, 'adapter': adapter_spec(setting),
                'base_study': {'ledger_sha256': original['ledger_sha256'],
                               **file_record(repository, repository / 'results/study/ledger.json')},
                'base_hyperparameters': dataclasses.asdict(config), 'policies': list(POLICIES),
                'training_inputs': data.ledger_inputs(), 'source_retrieval_validation': pool.ledger_inputs(),
                'assignment_files': receipt['assignments'],
                'assignment_receipt': file_record(repository, output / 'assignment_files.json'),
                'semantic_parity_audit': receipt.get('semantic_parity_audit'),
                'source_sha256': {name: sha256_file(repository / name) for name in code},
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'numpy': np.__version__, 'platform': platform.platform()},
                'checkpoint_serialization': 'torch.save_to_BytesIO_stable_archive_prefix',
                'execution_status': 'reconstructed_rerun_after_unsaved_prior_execution',
                'input_recovery': {'training_cache_matches_original_bytes': sha256_file(data.features_path) == original['identity']['inputs']['features']['sha256'],
                    'original_training_features_sha256': original['identity']['inputs']['features']['sha256'],
                    'actual_training_features_sha256': sha256_file(data.features_path),
                    'assignment_training_features_sha256': assignment_features_sha256,
                    'recovery_deviation': ('reencoded_training_features; original nonlinear control assignment bytes retained; prior unsaved fitted outcomes unavailable'
                        if setting == 'nonlinear' and sha256_file(data.features_path) != original['identity']['inputs']['features']['sha256'] else None)}}
    identity = json.loads(canonical_json(identity))
    return {'identity': identity, 'ledger_sha256': hashlib.sha256(canonical_json(identity)).hexdigest()}, assignments


def _checkpoint_bytes(checkpoint: dict) -> bytes:
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return stream.getvalue()


def _atomic_bytes(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.tmp-', suffix='.pt')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _best_epoch(history: list[dict], selector: str):
    return min(history, key=lambda row: (-row['validation'][selector]['score'], row['epoch']))['epoch']


def _completed(directory: Path, ledger: dict, policy: str, rate: float, seed: int):
    path = directory / 'completion.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    expected = {'ledger_sha256': ledger['ledger_sha256'], 'method': policy,
                'learning_rate': rate, 'seed': seed, 'epochs_completed': 10, 'checkpoint_digest_count': 11}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Incompatible replication completion: {directory}')
    if set(record.get('checkpoint_sha256', {})) != {f'epochs/epoch_{e:02d}.pt' for e in range(11)}:
        raise ValueError('Replication completion must retain all eleven checkpoint digests.')
    required = {'history.json'} | {f'validation/epoch_{e:02d}.npz' for e in range(11)} | set(record.get('retained_checkpoints', []))
    if set(record.get('artifact_sha256', {})) != required:
        raise ValueError('Incomplete replication retained-artifact inventory.')
    for relative, digest in record['artifact_sha256'].items():
        if not (directory / relative).exists() or sha256_file(directory / relative) != digest:
            raise ValueError(f'A replication artifact is missing or modified: {directory / relative}')
    history = json.loads((directory / 'history.json').read_text())
    if any(history.get(k) != v for k, v in expected.items() if k not in ('epochs_completed', 'checkpoint_digest_count')):
        raise ValueError('Replication history candidate identity differs.')
    rows = history['history']
    if [row['epoch'] for row in rows] != list(range(11)):
        raise ValueError('Replication requires each epoch 0 through 10 exactly once.')
    retained = {f'epochs/epoch_{e:02d}.pt' for e in (10, *[_best_epoch(rows, s) for s in SELECTORS])}
    if set(record['retained_checkpoints']) != retained:
        raise ValueError('Retained weights must be exactly both development winners and epoch10.')
    for row in rows:
        relative = f"epochs/epoch_{row['epoch']:02d}.pt"
        if row['checkpoint'] != relative or row['checkpoint_sha256'] != record['checkpoint_sha256'][relative]:
            raise ValueError('Epoch checkpoint digest binding is invalid.')
        for selector in SELECTORS:
            if not np.isfinite(row['validation'][selector]['score']):
                raise ValueError('Development selection scores must be finite.')
        if relative in retained and record['artifact_sha256'][relative] != record['checkpoint_sha256'][relative]:
            raise ValueError('Retained checkpoint and serialized epoch digests differ.')
    return record


def fit_replication_candidate(repository: Path, output: Path, config: StudyConfig, data: FeatureDataset,
                              pool: SourceRetrievalPool, ledger: dict, assignments: dict,
                              policy: str, rate: float, seed: int):
    if policy not in POLICIES or rate not in config.learning_rates or seed not in config.seeds:
        raise ValueError('Candidate is outside the frozen replication grid.')
    if policy in POLICIES[2:] and policy not in assignments:
        raise ValueError('Random control requires its frozen assignment.')
    directory = candidate_path(output, policy, rate, seed)
    with file_lock(directory / '.candidate.lock'):
        complete = _completed(directory, ledger, policy, rate, seed)
        if complete is not None:
            return {**complete, 'skipped_compatible_complete': True}
        if (directory / 'history.json').exists():
            if json.loads((directory / 'history.json').read_text()).get('ledger_sha256') != ledger['ledger_sha256']:
                raise ValueError('Incompatible incomplete replication candidate.')
        started = time.monotonic()
        seed_everything(seed, config.threads)
        spec = ledger['identity']['adapter']
        adapter = make_adapter(**spec)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=rate, weight_decay=config.weight_decay,
            betas=config.adam_betas, eps=config.adam_epsilon, amsgrad=config.adam_amsgrad,
            foreach=config.adam_foreach, fused=config.adam_fused)
        order_rng = torch.Generator(device='cpu').manual_seed(seed)
        loss_rng = torch.Generator(device='cpu').manual_seed(seed + 1_000_003)
        condition, draw, assignment_seed = policy_parts(policy)
        training_indices = data.split_indices['train']
        history, checkpoint_hashes, prediction_hashes = [], {}, {}
        for epoch in range(config.epochs + 1):
            epoch_started = time.monotonic()
            total_loss, total_images, steps = 0., 0, 0
            order_digest = None
            if epoch:
                adapter.train()
                order = torch.randperm(len(training_indices), generator=order_rng).tolist()
                ordered_indices = [training_indices[k] for k in order]
                order_digest = hashlib.sha256(np.asarray(ordered_indices, dtype='<i8').tobytes()).hexdigest()
                for offset in range(0, len(order), config.image_batch_size):
                    indices = ordered_indices[offset:offset + config.image_batch_size]
                    images, texts, relations, text_indices = data.batch(indices)
                    if policy in assignments:
                        relations = apply_promotion_assignment(relations, indices, text_indices, assignments[policy])
                    ai, at = adapter(images, texts)
                    loss = contrastive_loss(ai, at, relations, 'clip' if policy == 'source' else 'multipositive',
                        data.logit_scale, generator=loss_rng, semantic_text_features=texts, semantic_image_features=images,
                        semantic_threshold=config.semantic_threshold, semantic_softness=config.semantic_softness,
                        negative_weight=config.negative_weight, hardening_weight=config.hardening_weight,
                        smoothing=config.smoothing)
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite replication training loss.')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm, error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(indices)
                    total_images += len(indices)
                    steps += 1
            native = validate_adapter(adapter, data)
            source, predictions = validate_source_retrieval(adapter, pool)
            native_rows = native['per_image']
            predictions.update(native_image_ids=np.asarray([r['image_id'] for r in native_rows]),
                native_winning_text_ids=np.asarray([r['winning_text_id'] for r in native_rows]),
                native_retrieval_correct=np.asarray([r['known_positive_i2t_r1'] for r in native_rows]),
                native_relation_accuracy=np.asarray([r['relation_accuracy'] if r['relation_accuracy'] is not None else np.nan for r in native_rows]))
            relative = f'epochs/epoch_{epoch:02d}.pt'
            prediction_path = f'validation/epoch_{epoch:02d}.npz'
            metrics = {'native': native, 'source_retrieval': source}
            checkpoint = {'schema_version': 1, 'adapter': spec, 'state_dict': adapter.state_dict(),
                'ledger_sha256': ledger['ledger_sha256'], 'protocol_sha256': ledger['identity']['protocol']['sha256'],
                'method': policy, 'condition': condition, 'draw_id': draw, 'assignment_seed': assignment_seed,
                'learning_rate': rate, 'seed': seed, 'epoch': epoch,
                'state_id': state_id(policy, rate, seed, epoch),
                'validation': {k: {a: b for a, b in v.items() if a != 'per_image'} for k, v in metrics.items()},
                'base_hyperparameters': dataclasses.asdict(config)}
            serialized = _checkpoint_bytes(checkpoint)
            _atomic_bytes(directory / relative, serialized)
            checkpoint_hashes[relative] = hashlib.sha256(serialized).hexdigest()
            _atomic_npz(directory / prediction_path, **predictions)
            prediction_hashes[prediction_path] = sha256_file(directory / prediction_path)
            history.append({'epoch': epoch, 'mean_training_loss': total_loss / total_images if total_images else None,
                'training_images': total_images, 'gradient_steps': steps, 'batch_order_sha256': order_digest,
                'validation': metrics, 'development_predictions': prediction_path,
                'checkpoint': relative, 'checkpoint_sha256': checkpoint_hashes[relative],
                'seconds': time.monotonic() - epoch_started})
            atomic_json(directory / 'history.json', {'schema_version': 1, 'ledger_sha256': ledger['ledger_sha256'],
                'method': policy, 'condition': condition, 'draw_id': draw, 'learning_rate': rate,
                'seed': seed, 'adapter': spec, 'history': history})
        retained_epochs = sorted({config.epochs, *[_best_epoch(history, selector) for selector in SELECTORS]})
        retained = [f'epochs/epoch_{epoch:02d}.pt' for epoch in retained_epochs]
        for relative in checkpoint_hashes:
            if relative not in retained:
                (directory / relative).unlink()
        hashes = {**prediction_hashes, **{name: checkpoint_hashes[name] for name in retained},
                  'history.json': sha256_file(directory / 'history.json')}
        complete = {'schema_version': 1, 'ledger_sha256': ledger['ledger_sha256'], 'method': policy,
            'condition': condition, 'draw_id': draw, 'assignment_seed': assignment_seed,
            'learning_rate': rate, 'seed': seed, 'epochs_completed': config.epochs,
            'checkpoint_digest_count': len(checkpoint_hashes), 'checkpoint_sha256': checkpoint_hashes,
            'retained_checkpoints': retained, 'artifact_sha256': hashes,
            'best_epochs': {selector: _best_epoch(history, selector) for selector in SELECTORS},
            'seconds': time.monotonic() - started}
        atomic_json(directory / 'completion.json', complete)
        return complete


def select_and_export(repository: Path, output: Path, config: StudyConfig, ledger: dict):
    histories, states = {}, []
    for policy in POLICIES:
        condition, draw, assignment_seed = policy_parts(policy)
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_path(output, policy, rate, seed)
                completion = _completed(directory, ledger, policy, rate, seed)
                if completion is None:
                    raise ValueError(f'Selection requires all 45 completed replication candidates; missing {directory}')
                rows = json.loads((directory / 'history.json').read_text())['history']
                histories[(policy, rate, seed)] = rows
                for row in rows:
                    states.append({'state_id': state_id(policy, rate, seed, row['epoch']),
                        'condition': condition, 'method': policy, 'draw_id': draw, 'assignment_seed': assignment_seed,
                        'learning_rate': rate, 'seed': seed, 'epoch': row['epoch'], 'adapter': ledger['identity']['adapter'],
                        'checkpoint': str((directory / row['checkpoint']).relative_to(repository)),
                        'checkpoint_sha256': row['checkpoint_sha256'],
                        'checkpoint_retained': row['checkpoint'] in completion['retained_checkpoints'],
                        'validation': {k: {a: b for a, b in v.items() if a != 'per_image'} for k, v in row['validation'].items()}})
    if len(histories) != 45 or len(states) != 495:
        raise ValueError('Replication selection requires exactly 45 candidates and 495 epoch records.')
    selections = []
    for selector in SELECTORS:
        selected = {'schema_version': 1, 'selector': selector, 'ledger_sha256': ledger['ledger_sha256'],
                    'protocol_sha256': ledger['identity']['protocol']['sha256'], 'methods': {},
                    'test_outcomes_used': False, 'draw_selection': 'separate_no_winner_draw'}
        for policy in POLICIES:
            condition, draw, assignment_seed = policy_parts(policy)
            rate_rows = []
            for rate in config.learning_rates:
                best = {seed: min(histories[(policy, rate, seed)], key=lambda r: (-r['validation'][selector]['score'], r['epoch'])) for seed in config.seeds}
                rate_rows.append({'learning_rate': rate, 'mean_validation_score': float(np.mean([r['validation'][selector]['score'] for r in best.values()])), 'best': best})
            chosen = min(rate_rows, key=lambda r: (-r['mean_validation_score'], r['learning_rate']))
            runs = []
            for seed, row in chosen['best'].items():
                selection = {'selector': selector, 'condition': condition, 'method': policy, 'draw_id': draw,
                    'assignment_seed': assignment_seed, 'learning_rate': chosen['learning_rate'], 'seed': seed,
                    'epoch': row['epoch'], 'state_id': state_id(policy, chosen['learning_rate'], seed, row['epoch'])}
                selections.append(selection)
                runs.append({**selection, 'validation': {a: b for a, b in row['validation'][selector].items() if a != 'per_image'}})
            selected['methods'][policy] = {'condition': condition, 'draw_id': draw, 'assignment_seed': assignment_seed,
                'learning_rate': chosen['learning_rate'], 'mean_validation_score': chosen['mean_validation_score'],
                'learning_rate_scores': [{k: v for k, v in r.items() if k != 'best'} for r in rate_rows], 'runs': runs}
        atomic_json(output / f'selection_{selector}.json', selected)
    common = {'schema_version': 1, 'setting': ledger['identity']['setting'], 'adapter': ledger['identity']['adapter'],
              'ledger_sha256': ledger['ledger_sha256'], 'protocol_sha256': ledger['identity']['protocol']['sha256'],
              'candidate_count': 45, 'primary_epoch': 10, 'test_outcomes_used_for_selection': False,
              'draw_aggregation': 'aggregate_all_three_draws_without_winner_selection'}
    full_manifest = {**common, 'states': states, 'state_count': len(states), 'selections': selections}
    atomic_json(output / 'state_manifest.json', full_manifest)
    chosen_ids = {s['state_id'] for s in selections if s['epoch'] > 0}
    chosen = [s for s in states if s['epoch'] == 10 or s['state_id'] in chosen_ids]
    if any(not s['checkpoint_retained'] for s in chosen):
        raise ValueError('Evaluation selected a checkpoint that was not retained.')
    frozen = {'state_id': 'frozen', 'condition': 'frozen', 'method': 'frozen', 'draw_id': None,
              'assignment_seed': None, 'learning_rate': None, 'seed': None, 'epoch': 0,
              'checkpoint': None, 'checkpoint_sha256': None, 'adapter': ledger['identity']['adapter']}
    normalized_selections = [{**s, 'candidate_state_id': s['state_id'],
                             'state_id': 'frozen' if s['epoch'] == 0 else s['state_id']} for s in selections]
    manifest = {**common, 'states': [frozen] + chosen, 'state_count': len(chosen) + 1,
                'selections': normalized_selections, 'full_state_manifest': file_record(repository, output / 'state_manifest.json')}
    atomic_json(output / 'evaluation_manifest.json', manifest)
    return manifest
