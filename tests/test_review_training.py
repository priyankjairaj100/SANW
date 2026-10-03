"""Follow-up protocol, deterministic replay, all-epoch storage and selection tests."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gcr.adapters import ResidualAdapter
from gcr.review_controls import make_promotion_assignment
from gcr.review_training import (
    POLICIES, _completed, candidate_path, fit_review_candidate,
    policy_parts, select_and_export, validate_source_retrieval, verify_protocol,
)
from gcr.training import StudyConfig, atomic_json, fit_candidate, sha256_file
from test_training import fixture_dataset

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY / 'docs/REVIEW_FOLLOWUP_PROTOCOL.json'
PROTOCOL_SHA = '3edeae2f741842f6f23f19722b5aff892991457cc13e20e1b36179ed4db0bb34'


def tiny_pool():
    return SimpleNamespace(images=torch.eye(8)[:2], texts=torch.eye(8)[:2],
                           owner=torch.tensor([0, 1]), image_ids=['dev:a', 'dev:b'],
                           text_ids=['caption:a', 'caption:b'])


def test_frozen_protocol_schema_and_complete_inherited_config():
    original = json.loads((REPOSITORY / 'results/study/ledger.json').read_text())
    assert verify_protocol(PROTOCOL, PROTOCOL_SHA, original, StudyConfig())['training']['candidate_count'] == 72
    for changed in ({'learning_rates': (1e-4,)}, {'epochs': 5}, {'smoothing': .2}, {'threads': 1}, {'adam_foreach': True}):
        with pytest.raises(ValueError, match='disagrees'):
            verify_protocol(PROTOCOL, PROTOCOL_SHA, original, dataclasses.replace(StudyConfig(), **changed))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify_protocol(PROTOCOL, '0' * 64, original, StudyConfig())


def test_source_validation_uses_ownership_and_stable_manifest_ties():
    # Every score ties. First caption belongs to image 1; first image is image 0.
    pool = SimpleNamespace(images=torch.tensor([[1., 0.], [1., 0.]]),
                           texts=torch.tensor([[1., 0.], [1., 0.], [1., 0.]]),
                           owner=torch.tensor([1, 0, 1]), image_ids=['a', 'b'], text_ids=['x', 'y', 'z'])
    summary, raw = validate_source_retrieval(ResidualAdapter(2), pool)
    assert summary['i2t_r1'] == .5
    assert summary['t2i_r1'] == 1 / 3
    assert summary['score'] == (.5 + 1 / 3) / 2
    assert raw['image_correct'].tolist() == [False, True]
    assert raw['text_correct'].tolist() == [False, True, False]
    assert raw['image_winning_text_index'].tolist() == [0, 0]
    assert raw['text_winning_image_index'].tolist() == [0, 0, 0]


@pytest.mark.parametrize('policy,original_method', [('source', 'clip'), ('supported', 'multipositive')])
def test_all_epoch_replay_matches_original_and_resume_verifies_every_artifact(tmp_path, policy, original_method):
    data = fixture_dataset(tmp_path / 'data')
    config = StudyConfig(feature_dim=8, image_batch_size=1, threads=1)
    fit_candidate(tmp_path, tmp_path / 'results/study', data, config, {'ledger_sha256': 'original'}, original_method, 1e-4, 17)
    ledger = {'ledger_sha256': 'new', 'identity': {'protocol': {'sha256': PROTOCOL_SHA}}}
    output = tmp_path / 'followup'
    result = fit_review_candidate(tmp_path, output, config, data, tiny_pool(), ledger, {}, policy, 1e-4, 17)
    parity = result['replay_parity']
    assert parity['checked_epochs'] == 11
    assert parity['native_validation_exact'] and parity['training_loss_exact'] and parity['best_state_exact']
    directory = candidate_path(output, policy, 1e-4, 17)
    assert len(list((directory / 'epochs').glob('*.pt'))) == 11
    assert len(list((directory / 'validation').glob('*.npz'))) == 11
    assert result['checkpoint_count'] == 11
    assert fit_review_candidate(tmp_path, output, config, data, tiny_pool(), ledger, {}, policy, 1e-4, 17)['skipped_compatible_complete']
    (directory / 'epochs/epoch_05.pt').write_bytes(b'modified')
    with pytest.raises(ValueError, match='missing or modified'):
        fit_review_candidate(tmp_path, output, config, data, tiny_pool(), ledger, {}, policy, 1e-4, 17)


def test_random_control_requires_assignment_and_changes_training(tmp_path):
    data = fixture_dataset(tmp_path / 'data')
    config = StudyConfig(feature_dim=8, image_batch_size=2, threads=1)
    ledger = {'ledger_sha256': 'new', 'identity': {'protocol': {'sha256': PROTOCOL_SHA}}}
    with pytest.raises(ValueError, match='requires its frozen'):
        fit_review_candidate(tmp_path, tmp_path / 'review', config, data, tiny_pool(), ledger, {}, 'count_only_draw_0', 1e-3, 17)
    assignment = make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts, data.pairs,
        data.split_indices['train'], 'count_only', 101, {key: 'a' * 64 for key in ('manifest_sha256', 'features_sha256', 'protocol_sha256')},
        text_strings=[row['text'] for row in data.manifest['texts']])
    output = tmp_path / 'review'
    result = fit_review_candidate(tmp_path, output, config, data, tiny_pool(), ledger, {'count_only_draw_0': assignment}, 'count_only_draw_0', 1e-3, 17)
    assert result['replay_parity']['required'] is False
    paths = candidate_path(output, 'count_only_draw_0', 1e-3, 17) / 'epochs'
    initial = torch.load(paths / 'epoch_00.pt', weights_only=True)
    final = torch.load(paths / 'epoch_10.pt', weights_only=True)
    assert final['assignment_seed'] == 101 and final['draw_id'] == 0 and final['condition'] == 'count_only'
    assert any(not torch.equal(initial['state_dict'][name], value) for name, value in final['state_dict'].items())


def synthetic_grid(root):
    config, output = StudyConfig(), root / 'review'
    ledger = {'ledger_sha256': 'synthetic', 'identity': {'protocol': {'sha256': PROTOCOL_SHA}}}
    for policy in POLICIES:
        _, draw, _ = policy_parts(policy)
        preferred_rate = config.learning_rates[draw] if draw is not None else 1e-4
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_path(output, policy, rate, seed)
                hashes, history = {}, []
                for epoch in range(11):
                    relative = f'epochs/epoch_{epoch:02d}.pt'
                    path = directory / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f'{policy},{rate},{seed},{epoch}'.encode())
                    hashes[relative] = sha256_file(path)
                    pred = directory / f'validation/epoch_{epoch:02d}.npz'
                    pred.parent.mkdir(parents=True, exist_ok=True)
                    pred.write_bytes(b'synthetic-predictions')
                    hashes[str(pred.relative_to(directory))] = sha256_file(pred)
                    # Native all ties: shared low LR and epoch0. Source selector
                    # tests draw-specific LR selection, with equal epoch1/2 maxima.
                    source = .8 if rate == preferred_rate and epoch in (1, 2) else .2
                    history.append({'epoch': epoch, 'checkpoint': relative, 'checkpoint_sha256': hashes[relative],
                                    'validation': {'native': {'score': .5}, 'source_retrieval': {'score': source}}})
                atomic_json(directory / 'history.json', {'ledger_sha256': 'synthetic', 'method': policy,
                    'learning_rate': rate, 'seed': seed, 'history': history})
                hashes['history.json'] = sha256_file(directory / 'history.json')
                atomic_json(directory / 'completion.json', {'ledger_sha256': 'synthetic', 'method': policy,
                    'learning_rate': rate, 'seed': seed, 'epochs_completed': 10, 'checkpoint_count': 11, 'artifact_sha256': hashes})
    atomic_json(root / 'results/study/selection.json', {'methods': {method: {'learning_rate': 1e-4,
        'runs': [{'seed': seed, 'epoch': 0} for seed in config.seeds]} for method in ('clip', 'multipositive')}})
    return config, output, ledger


def test_full_grid_separate_draws_selection_ties_and_no_weight_copies(tmp_path):
    config, output, ledger = synthetic_grid(tmp_path)
    manifest = select_and_export(tmp_path, output, config, ledger)
    assert manifest['state_count'] == len({row['state_id'] for row in manifest['states']}) == 792
    assert len(manifest['selections']) == 48
    for row in manifest['selections']:
        if row['selector'] == 'native':
            assert row['learning_rate'] == 1e-4 and row['epoch'] == 0
        elif row['draw_id'] is not None:
            assert row['learning_rate'] == config.learning_rates[row['draw_id']] and row['epoch'] == 1
    assert len(list(output.rglob('*.pt'))) == 792
    candidate_path(output, 'count_only_draw_0', 1e-4, 17).joinpath('completion.json').unlink()
    with pytest.raises(ValueError, match='all 72 completed'):
        select_and_export(tmp_path, output, config, ledger)


def test_truncated_receipt_is_not_complete(tmp_path):
    config, output, ledger = synthetic_grid(tmp_path)
    directory = candidate_path(output, 'source', 1e-4, 17)
    receipt = json.loads((directory / 'completion.json').read_text())
    receipt['artifact_sha256'].pop('epochs/epoch_05.pt')
    atomic_json(directory / 'completion.json', receipt)
    with pytest.raises(ValueError, match='all eleven'):
        _completed(directory, ledger, 'source', 1e-4, 17)


def test_selection_rejects_duplicate_epoch_history_even_with_updated_receipt(tmp_path):
    config, output, ledger = synthetic_grid(tmp_path)
    directory = candidate_path(output, 'source', 1e-4, 17)
    history = json.loads((directory / 'history.json').read_text())
    history['history'][5]['epoch'] = 4
    atomic_json(directory / 'history.json', history)
    receipt = json.loads((directory / 'completion.json').read_text())
    receipt['artifact_sha256']['history.json'] = sha256_file(directory / 'history.json')
    atomic_json(directory / 'completion.json', receipt)
    with pytest.raises(ValueError, match='each epoch 0 through 10'):
        select_and_export(tmp_path, output, config, ledger)
