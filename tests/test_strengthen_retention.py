"""Objective identities, detached KL domain, interpolation and selection checks."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gcr.adapters import ResidualAdapter
from gcr.losses import contrastive_loss
from gcr.retention_losses import directional_loss, retention_loss, source_distribution_kl
from gcr.strengthen_retention import (ALPHAS, POLICIES, PROTOCOL_SHA256, choose_family,
    feasible, fit_candidate, fit_matched_controls, scaled_adapter, select_and_export, update_norm, verify_protocol)
from gcr.review_controls import make_promotion_assignment
from gcr.training import StudyConfig
from test_training import fixture_dataset

REPOSITORY = Path(__file__).resolve().parents[1]


def tiny_pool():
    return SimpleNamespace(images=torch.eye(8)[:2], texts=torch.eye(8)[:2], owner=torch.tensor([0, 1]),
                           image_ids=['dev:a', 'dev:b'], text_ids=['caption:a', 'caption:b'])


def fixture_features():
    gen = torch.Generator().manual_seed(7)
    images = F.normalize(torch.randn(3, 8, generator=gen), dim=-1)
    texts = F.normalize(torch.randn(9, 8, generator=gen), dim=-1)
    rel = torch.tensor([[1, 1, 2, 3, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 2, 3, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 2]])
    return images, texts, rel


def test_frozen_protocol_checks_schedule_and_exact_hash():
    path = REPOSITORY / 'results/strengthen_retention/protocol_v3.json'
    assert verify_protocol(path, StudyConfig())['evaluation']['primary_family_size'] == 80
    with pytest.raises(ValueError, match='Configuration differs'):
        verify_protocol(path, dataclasses.replace(StudyConfig(), epochs=5))


@pytest.mark.parametrize('policy,original', [('source', 'clip'), ('supported', 'multipositive')])
def test_native_objective_and_gradients_replay_exactly(policy, original):
    images, texts, rel = fixture_features()
    images.requires_grad_()
    texts.requires_grad_()
    old = contrastive_loss(images, texts, rel, original, 7.)
    new = retention_loss(images, texts, rel, policy, 7.)
    assert torch.equal(old, new)
    old_grad = torch.autograd.grad(old, (images, texts), retain_graph=True)
    new_grad = torch.autograd.grad(new, (images, texts))
    assert all(torch.equal(a, b) for a, b in zip(old_grad, new_grad))


def test_directional_factorial_uses_full_candidate_matrix_and_fixed_half_weights():
    images, texts, rel = fixture_features()
    logits = (images @ texts.T) * 7.
    source = directional_loss(logits, rel, image_expanded=False, reverse_expanded=False)
    support = directional_loss(logits, rel, image_expanded=True, reverse_expanded=True)
    image = retention_loss(images, texts, rel, 'image_source_only', 7.)
    reverse = retention_loss(images, texts, rel, 'reverse_source_only', 7.)
    assert torch.equal(source, contrastive_loss(images, texts, rel, 'clip', 7.))
    assert torch.equal(support, contrastive_loss(images, texts, rel, 'multipositive', 7.))
    torch.testing.assert_close(source + support, image + reverse)
    # An unpromoted hypothesis remains in the image-query denominator.
    altered = logits.clone()
    altered[0, 3] += 100
    assert directional_loss(altered, rel, image_expanded=False, reverse_expanded=True) > image


def test_allocation_is_declared_convex_objective_mix():
    images, texts, rel = fixture_features()
    for mix in (.5, .8):
        actual = retention_loss(images, texts, rel, f'allocation_{mix}', 5.)
        expected = mix * retention_loss(images, texts, rel, 'source', 5.) + (1-mix) * retention_loss(images, texts, rel, 'supported', 5.)
        assert torch.equal(actual, expected)


def test_kl_exact_zero_detaches_teacher_and_excludes_hypotheses_both_directions():
    images, texts, rel = fixture_features()
    teacher = (images @ texts.T).requires_grad_()
    student = teacher.detach().clone().requires_grad_()
    image_kl, text_kl = source_distribution_kl(student, teacher, rel)
    assert image_kl.item() == text_kl.item() == 0.
    perturbed = student.detach().clone()
    perturbed[:, ~(rel == 1).any(0)] += 1000
    assert all(x.item() == 0. for x in source_distribution_kl(perturbed, teacher, rel))
    changed = student.detach().clone()
    changed[0, 0] += 3
    changed.requires_grad_()
    penalty = sum(source_distribution_kl(changed, teacher, rel))
    assert penalty.item() > 0
    penalty.backward()
    assert teacher.grad is None
    assert torch.equal(changed.grad[:, ~(rel == 1).any(0)], torch.zeros_like(changed.grad[:, ~(rel == 1).any(0)]))
    assert changed.grad[:, (rel == 1).any(0)].abs().sum() > 0


def test_teacher_temperature_square_and_beta_factor_match_manual_calculation():
    images, texts, rel = fixture_features()
    model = ResidualAdapter(8)
    with torch.no_grad():
        model.image.weight[0, 1] = .2
    adapted_images, adapted_texts = model(images, texts)
    teacher = (F.normalize(images, dim=-1) @ F.normalize(texts, dim=-1).T) * 7
    student = (adapted_images @ adapted_texts.T) * 7
    ki, kt = source_distribution_kl(student, teacher, rel, 2.)
    base = retention_loss(adapted_images, adapted_texts, rel, 'supported', 7.)
    for beta in (1, 4, 16):
        actual = retention_loss(adapted_images, adapted_texts, rel, f'distilled_{beta}', 7., frozen_images=images, frozen_texts=texts)
        torch.testing.assert_close(actual, base + beta * 4 * .5 * (ki + kt), rtol=0, atol=0)


def test_wise_interpolates_both_parameters_before_normalization_and_zero_is_frozen():
    images, texts, _ = fixture_features()
    model = ResidualAdapter(8)
    with torch.no_grad():
        model.image.weight.copy_(torch.arange(64).reshape(8, 8) / 128)
        model.text.weight.copy_(torch.arange(64).reshape(8, 8).T / 64)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    for alpha in ALPHAS:
        scaled = scaled_adapter(model, alpha)
        torch.testing.assert_close(scaled.encode_image(images), F.normalize(images + alpha * model.image(images), dim=-1))
        torch.testing.assert_close(scaled.encode_text(texts), F.normalize(texts + alpha * model.text(texts), dim=-1))
        assert update_norm(scaled.state_dict()) == pytest.approx(alpha * update_norm(original))
    frozen = ResidualAdapter(8)
    assert all(torch.equal(a, b) for a, b in zip(scaled_adapter(model, 0)(images, texts), frozen(images, texts)))
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def row(epoch, relation, i2t=.8, t2i=.7):
    return {'epoch': epoch, 'native': {'relation_accuracy': relation},
            'source_retrieval': {'i2t_r1': i2t, 't2i_r1': t2i}}


def test_selector_both_constraints_epoch_zero_ties_and_common_hyperparameters():
    candidates = []
    for rate in (1e-4, 3e-4):
        for param in (1., 4.):
            histories = {seed: [row(0, .5), row(1, .9, t2i=.68), row(2, .7), row(3, .7)] for seed in (17,29,43)}
            candidates.append({'policy': f'distilled_{param:g}', 'learning_rate': rate, 'parameter': param, 'histories': histories})
    selected = choose_family(candidates, (17,29,43), 1.)
    assert selected['learning_rate'] == 1e-4 and selected['parameter'] == 1.
    assert all(r['epoch'] == 2 for r in selected['best'].values())
    assert selected['feasible_epochs'][17] == [0,2,3]
    assert feasible(row(1, .9, t2i=.69), row(0,.5), 1.)
    assert not feasible(row(1, .9, i2t=.78), row(0,.5), 1.)
    # A different optimal epoch is allowed for each seed, never a different LR.
    candidates[0]['histories'][17][2]['native']['relation_accuracy'] = .4
    candidates[0]['histories'][17][3]['native']['relation_accuracy'] = .4
    chosen = choose_family(candidates[:1], (17,29,43), 1.)
    assert chosen['best'][17]['epoch'] == 0
    assert chosen['best'][29]['epoch'] == 2


def test_fit_receipts_all_epoch_storage_and_resume_tamper_detection(tmp_path):
    data = fixture_dataset(tmp_path / 'data')
    config = StudyConfig(feature_dim=8, epochs=2, image_batch_size=2, threads=1)
    ledger = {'ledger_sha256': 'new', 'identity': {'encoder': 'vit_b32', 'inputs': {'logit_scale': 7.}}}
    output = tmp_path / 'retention'
    result = fit_candidate(tmp_path, output, config, data, tiny_pool(), ledger, 'distilled_4', 1e-3, 17)
    assert result['checkpoint_count'] == 3
    assert fit_candidate(tmp_path, output, config, data, tiny_pool(), ledger, 'distilled_4', 1e-3, 17)['skipped_compatible_complete']
    path = output / 'candidates/distilled_4/lr_0.001/seed_17/epochs/epoch_01.pt'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='missing or modified'):
        fit_candidate(tmp_path, output, config, data, tiny_pool(), ledger, 'distilled_4', 1e-3, 17)


def test_complete_tiny_grid_selects_and_exports_without_reading_any_test_data(tmp_path):
    data = fixture_dataset(tmp_path / 'data')
    config = StudyConfig(feature_dim=8, epochs=1, image_batch_size=2, threads=1)
    ledger = {'ledger_sha256': 'new', 'identity': {'encoder': 'vit_b32', 'inputs': {'logit_scale': data.logit_scale}}}
    output = tmp_path / 'retention'
    for policy in POLICIES:
        for rate in config.learning_rates:
            for seed in config.seeds:
                fit_candidate(tmp_path, output, config, data, tiny_pool(), ledger, policy, rate, seed)
    manifest = select_and_export(tmp_path, output, config, ledger)
    assert len(manifest['selections']) == 42
    assert manifest['test_outcomes_used_for_selection'] is False
    selection = json.loads((output / 'selection_primary.json').read_text())
    for family, selected in selection['families'].items():
        assert len(selected['runs']) == 3
        assert len({r['learning_rate'] for r in selected['runs']}) == 1
        assert selected['all_three_nonzero'] == all(r['epoch'] > 0 and r['update_norm'] > 0 for r in selected['runs'])
    assert select_and_export(tmp_path, output, config, ledger) == manifest
    assignments = {draw: make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
        data.pairs, data.split_indices['train'], 'score_stratified', seed,
        {key: 'a' * 64 for key in ('manifest_sha256', 'features_sha256', 'protocol_sha256')},
        text_strings=[r['text'] for r in data.manifest['texts']]) for draw, seed in enumerate((101,211,307))}
    final = fit_matched_controls(tmp_path, output, config, data, tiny_pool(), ledger, assignments)
    assert final['matched_controls_complete']
    matched = [r for r in final['states'] if r['family'] == 'matched_distilled']
    assert len(matched) == 9
    supported = {r['seed']: r for r in selection['families']['distilled']['runs']}
    for run in matched:
        assert run['learning_rate'] == supported[run['seed']]['learning_rate']
        assert run['epoch'] == supported[run['seed']]['epoch']
        assert run['beta'] == supported[run['seed']]['beta']
    assert select_and_export(tmp_path, output, config, ledger) == final
