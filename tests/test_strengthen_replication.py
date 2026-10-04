"""Architecture, paired RNG, retention, strict provenance and development selectors."""
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from gcr.strengthen_replication import (
    POLICIES, SELECTORS, NonlinearResidualAdapter, _best_epoch, _completed,
    _original_config, adapter_spec, fit_replication_candidate, load_replication_adapter,
    make_adapter, select_and_export, verify_replication_protocol,
)
from gcr.review_training import candidate_path, policy_parts
from gcr.training import StudyConfig, atomic_json, sha256_file
from test_review_training import tiny_pool
from test_training import fixture_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_prescribed_architectures_parameter_counts_identity_and_nonlinearity():
    torch.manual_seed(17)
    model = NonlinearResidualAdapter()
    assert sum(p.numel() for p in model.parameters()) == 262144
    assert model.image[0].bias is model.image[2].bias is None
    assert isinstance(model.image[1], torch.nn.ReLU)
    assert torch.count_nonzero(model.image[2].weight) == 0
    assert torch.count_nonzero(model.text[2].weight) == 0
    assert not torch.equal(model.image[0].weight, model.text[0].weight)
    x = F.normalize(torch.randn(7, 512), dim=-1)
    assert torch.equal(model.encode_image(x), F.normalize(x, dim=-1))
    assert torch.equal(model.encode_text(x), F.normalize(x, dim=-1))
    assert sum(p.numel() for p in make_adapter('linear', 1024).parameters()) == 2097152
    # Once output weights change, ReLU does not reduce to a linear odd map.
    with torch.no_grad():
        model.image[2].weight.fill_(0.01)
    assert not torch.allclose(model.encode_image(-x), -model.encode_image(x))


@pytest.mark.parametrize('setting,path', [('nonlinear', 'docs/STRENGTHEN_REPLICATION_PROTOCOL.json'),
                                         ('rn50', 'docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json')])
def test_protocol_inherits_all_base_hyperparameters(setting, path):
    original, base = _original_config(ROOT)
    config = dataclasses.replace(base, feature_dim=512 if setting == 'nonlinear' else 1024)
    protocol = ROOT / path
    assert verify_replication_protocol(protocol, sha256_file(protocol), setting, config, original)['training']['epochs'] == 10
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify_replication_protocol(protocol, '0' * 64, setting, config, original)
    for changes in ({'threads': 1}, {'weight_decay': 0.}, {'seeds': (17,)}, {'learning_rates': (.0001,)}):
        with pytest.raises(ValueError, match='inherit every'):
            verify_replication_protocol(protocol, sha256_file(protocol), setting, dataclasses.replace(config, **changes), original)


def tiny_ledger(kind='nonlinear'):
    return {'ledger_sha256': 'test-only-ledger', 'identity': {
        'setting': 'nonlinear' if kind == 'nonlinear' else 'rn50',
        'adapter': {'kind': kind, 'dimension': 8, 'bottleneck': 128},
        'protocol': {'sha256': 'a' * 64}}}


def test_paired_rng_all_epoch_hashes_recovery_retention_and_adapter_loader(tmp_path):
    data = fixture_dataset(tmp_path / 'data')
    config = StudyConfig(feature_dim=8, image_batch_size=1, threads=1)
    ledger = tiny_ledger()
    receipts, histories = {}, {}
    for policy in ('source', 'supported'):
        receipts[policy] = fit_replication_candidate(tmp_path, tmp_path / 'run', config, data, tiny_pool(), ledger, {}, policy, .001, 17)
        directory = candidate_path(tmp_path / 'run', policy, .001, 17)
        histories[policy] = json.loads((directory / 'history.json').read_text())['history']
        assert len(histories[policy]) == len(receipts[policy]['checkpoint_sha256']) == 11
        expected = {10, *[_best_epoch(histories[policy], selector) for selector in SELECTORS]}
        assert {int(p.stem[-2:]) for p in (directory / 'epochs').glob('*.pt')} == expected
        assert len(list((directory / 'validation').glob('*.npz'))) == 11
        checkpoint = torch.load(directory / 'epochs/epoch_10.pt', weights_only=True)
        model = load_replication_adapter(checkpoint)
        assert isinstance(model, NonlinearResidualAdapter)
        assert _completed(directory, ledger, policy, .001, 17) is not None
    assert [r['batch_order_sha256'] for r in histories['source']] == [r['batch_order_sha256'] for r in histories['supported']]
    # Policies share the exact initialization; independent loss gradients change learned output.
    assert receipts['source']['checkpoint_sha256']['epochs/epoch_10.pt'] != receipts['supported']['checkpoint_sha256']['epochs/epoch_10.pt']
    assert fit_replication_candidate(tmp_path, tmp_path / 'run', config, data, tiny_pool(), ledger, {}, 'source', .001, 17)['skipped_compatible_complete']
    # Fresh replay reproduces all serialized checkpoints, including pruned epochs.
    repeated = fit_replication_candidate(tmp_path, tmp_path / 'repeat', config, data, tiny_pool(), ledger, {}, 'source', .001, 17)
    assert repeated['checkpoint_sha256'] == receipts['source']['checkpoint_sha256']
    directory = candidate_path(tmp_path / 'run', 'source', .001, 17)
    (directory / 'epochs/epoch_10.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='missing or modified'):
        _completed(directory, ledger, 'source', .001, 17)


def test_assignment_required_and_all_eleven_digests_required(tmp_path):
    data = fixture_dataset(tmp_path / 'data')
    config = StudyConfig(feature_dim=8, image_batch_size=2, threads=1)
    ledger = tiny_ledger('linear')
    with pytest.raises(ValueError, match='frozen assignment'):
        fit_replication_candidate(tmp_path, tmp_path / 'run', config, data, tiny_pool(), ledger, {}, POLICIES[2], .001, 17)
    fit_replication_candidate(tmp_path, tmp_path / 'run', config, data, tiny_pool(), ledger, {}, 'source', .001, 17)
    directory = candidate_path(tmp_path / 'run', 'source', .001, 17)
    record = json.loads((directory / 'completion.json').read_text())
    del record['checkpoint_sha256']['epochs/epoch_04.pt']
    atomic_json(directory / 'completion.json', record)
    with pytest.raises(ValueError, match='all eleven'):
        _completed(directory, ledger, 'source', .001, 17)


def synthetic_grid(root):
    output, config, ledger = root / 'run', StudyConfig(), tiny_ledger()
    for policy in POLICIES:
        _, draw, _ = policy_parts(policy)
        preferred = config.learning_rates[draw] if draw is not None else .0001
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_path(output, policy, rate, seed)
                rows, digests = [], {}
                for epoch in range(11):
                    checkpoint = f'epochs/epoch_{epoch:02d}.pt'
                    payload = f'{policy},{rate},{seed},{epoch}'.encode()
                    digests[checkpoint] = hashlib.sha256(payload).hexdigest()
                    source = .8 if rate == preferred and epoch in (1, 2) else .2
                    rows.append({'epoch': epoch, 'checkpoint': checkpoint, 'checkpoint_sha256': digests[checkpoint],
                                 'validation': {'native': {'score': .5}, 'source_retrieval': {'score': source}}})
                retained_epochs = {10, *[_best_epoch(rows, selector) for selector in SELECTORS]}
                retained, artifacts = [], {}
                for epoch in retained_epochs:
                    relative = f'epochs/epoch_{epoch:02d}.pt'
                    path = directory / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f'{policy},{rate},{seed},{epoch}'.encode())
                    retained.append(relative)
                    artifacts[relative] = sha256_file(path)
                for epoch in range(11):
                    relative = f'validation/epoch_{epoch:02d}.npz'
                    path = directory / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'test-only predictions')
                    artifacts[relative] = sha256_file(path)
                atomic_json(directory / 'history.json', {'ledger_sha256': ledger['ledger_sha256'],
                    'method': policy, 'learning_rate': rate, 'seed': seed, 'history': rows})
                artifacts['history.json'] = sha256_file(directory / 'history.json')
                atomic_json(directory / 'completion.json', {'ledger_sha256': ledger['ledger_sha256'], 'method': policy,
                    'learning_rate': rate, 'seed': seed, 'epochs_completed': 10, 'checkpoint_digest_count': 11,
                    'checkpoint_sha256': digests, 'retained_checkpoints': retained, 'artifact_sha256': artifacts})
    return output, config, ledger


def test_complete_grid_earliest_lower_rate_ties_separate_draw_selection_and_frozen_dedup(tmp_path):
    output, config, ledger = synthetic_grid(tmp_path)
    manifest = select_and_export(tmp_path, output, config, ledger)
    assert manifest['candidate_count'] == 45
    assert len(manifest['selections']) == 30
    assert len([s for s in manifest['states'] if s['epoch'] == 10]) == 45
    assert len([s for s in manifest['states'] if s['state_id'] == 'frozen']) == 1
    assert len(manifest['states']) == len({s['state_id'] for s in manifest['states']}) == 61
    for selection in manifest['selections']:
        if selection['selector'] == 'native':
            assert selection['learning_rate'] == .0001 and selection['epoch'] == 0
            assert selection['state_id'] == 'frozen'
        else:
            assert selection['epoch'] == 1
            if selection['draw_id'] is not None:
                assert selection['learning_rate'] == config.learning_rates[selection['draw_id']]
    full = json.loads((output / 'state_manifest.json').read_text())
    assert full['state_count'] == 495
    candidate_path(output, 'source', .0001, 17).joinpath('completion.json').unlink()
    with pytest.raises(ValueError, match='all 45'):
        select_and_export(tmp_path, output, config, ledger)


def test_recovered_control_audit_rejects_changed_strata_or_targets_but_reports_roundoff(tmp_path):
    import copy
    from gcr.review_controls import make_promotion_assignment
    from gcr.strengthen_replication import audit_assignment_semantics
    data = fixture_dataset(tmp_path / 'data')
    assignment = make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
        data.pairs, data.split_indices['train'], 'score_stratified', 101,
        {key: 'a' * 64 for key in ('manifest_sha256', 'features_sha256', 'protocol_sha256')},
        [row['text'] for row in data.manifest['texts']])
    roundoff = copy.deepcopy(assignment)
    roundoff['records'][0]['hypotheses'][0]['frozen_cosine'] += 1e-9
    audit = audit_assignment_semantics(assignment, roundoff)
    assert audit['passed'] and audit['max_absolute_frozen_cosine_error'] > 0
    changed = copy.deepcopy(assignment)
    changed['records'][0]['score_bins'][0]['hypothesis_text_indices'].reverse()
    assert not audit_assignment_semantics(assignment, changed)['passed']
    changed = copy.deepcopy(assignment)
    changed['records'][0]['promoted_text_ids'][0] += 'changed'
    assert not audit_assignment_semantics(assignment, changed)['passed']
