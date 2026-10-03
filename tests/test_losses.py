"""Objective checks against hand calculations, finite differences and invariants."""
import math

import pytest
import torch
from torch.nn import functional as F

from gcr.adapters import ResidualAdapter
from gcr.losses import (
    METHODS, Policy, contrastive_loss, pairwise_ranking_loss, policy_components,
    symmetric_weighted_loss, weighted_row_loss,
)

DTYPE = torch.float64


def relation_fixture():
    return torch.tensor([[1, 1, 2, 3, 4, 0, 0], [0, 0, 0, 0, 0, 1, 3]])


def features_fixture():
    gen = torch.Generator().manual_seed(71)
    images = F.normalize(torch.randn(2, 4, generator=gen, dtype=DTYPE), dim=1)
    texts = F.normalize(torch.randn(7, 4, generator=gen, dtype=DTYPE), dim=1)
    return images, texts


def test_single_positive_loss_matches_cross_entropy_and_transpose():
    logits = torch.tensor([[1.2, -0.4], [0.8, 1.6]], dtype=DTYPE)
    policy = Policy(torch.eye(2, dtype=torch.bool), torch.ones_like(logits), 0.)
    expected = (F.cross_entropy(logits, torch.arange(2)) + F.cross_entropy(logits.T, torch.arange(2))) / 2
    torch.testing.assert_close(symmetric_weighted_loss(logits, policy), expected)
    torch.testing.assert_close(symmetric_weighted_loss(logits.T, Policy(policy.positives.T, policy.weights.T, 0.)), expected)


def test_rectangular_loss_normalizes_each_direction_and_skips_unanchored_texts():
    logits = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=DTYPE)
    positive = torch.tensor([[True, True, False, False], [False, False, True, False]])
    loss = symmetric_weighted_loss(logits, Policy(positive, torch.ones_like(logits), 0.))
    image_expected = ((torch.logsumexp(logits[0], 0) - 1.5) + (torch.logsumexp(logits[1], 0) - 7)) / 2
    # Column 3 has no positive and is not a reverse-direction anchor.
    text_expected = ((torch.logsumexp(logits[:, 0], 0) - 1)
                     + (torch.logsumexp(logits[:, 1], 0) - 2)
                     + (torch.logsumexp(logits[:, 2], 0) - 7)) / 3
    torch.testing.assert_close(loss, (image_expected + text_expected) / 2)


def test_weighted_gradient_matches_closed_form_and_finite_difference():
    logits = torch.tensor([[0.2, -0.4, 1.2, 50.]], dtype=DTYPE, requires_grad=True)
    positives = torch.tensor([[True, True, False, False]])
    weights = torch.tensor([[1., 1., 0.3, 0.]], dtype=DTYPE, requires_grad=True)
    loss = weighted_row_loss(logits, positives, weights)
    gradient, weight_gradient = torch.autograd.grad(loss, (logits, weights), allow_unused=True)
    assert weight_gradient is None
    masses = weights.detach() * torch.exp(logits.detach())
    expected = masses / masses.sum() - positives.to(DTYPE) / 2
    torch.testing.assert_close(gradient, expected, atol=1e-13, rtol=1e-13)
    eps = 1e-6
    numeric = torch.zeros_like(logits)
    for k in range(logits.numel()):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[0, k] += eps
        minus[0, k] -= eps
        numeric[0, k] = (weighted_row_loss(plus, positives, weights) - weighted_row_loss(minus, positives, weights)) / (2 * eps)
    torch.testing.assert_close(gradient, numeric, atol=2e-9, rtol=2e-9)
    assert gradient[0, 3].item() == 0
    assert weights.grad is None


def test_stability_with_extreme_logits_and_exact_zero_mask():
    logits = torch.tensor([[1000., -1000., 100000.]], dtype=DTYPE, requires_grad=True)
    loss = weighted_row_loss(logits, torch.tensor([[True, False, False]]), torch.tensor([[1., .25, 0.]], dtype=DTYPE))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0, 2].item() == 0


def test_positive_expansion_changes_target_without_changing_denominator():
    logits = torch.tensor([[.1, .8, -.2, 1.7]], dtype=DTYPE, requires_grad=True)
    weights = torch.ones_like(logits)
    initial = torch.tensor([[True, False, False, False]])
    expanded = torch.tensor([[True, True, True, False]])
    original_grad, = torch.autograd.grad(weighted_row_loss(logits, initial, weights), logits)
    expanded_grad, = torch.autograd.grad(weighted_row_loss(logits, expanded, weights), logits)
    torch.testing.assert_close(expanded_grad - original_grad, torch.tensor([[2/3, -1/3, -1/3, 0.]], dtype=DTYPE))


def test_normalizer_matched_constant_preserves_positive_and_total_negative_gradients():
    logits = torch.tensor([[.4, -.3, .8, 1.2]], dtype=DTYPE, requires_grad=True)
    positive = torch.tensor([[True, True, False, False]])
    weights = torch.tensor([[1., 1., .2, .8]], dtype=DTYPE)
    masses = logits.detach().exp()[~positive]
    constant = (masses * weights[~positive]).sum() / masses.sum()
    matched = torch.where(positive, torch.ones_like(weights), constant)
    a = weighted_row_loss(logits, positive, weights)
    b = weighted_row_loss(logits, positive, matched)
    torch.testing.assert_close(a, b)
    ga, = torch.autograd.grad(a, logits)
    gb, = torch.autograd.grad(b, logits)
    torch.testing.assert_close(ga[positive], gb[positive])
    torch.testing.assert_close(ga[~positive].sum(), gb[~positive].sum())
    assert not torch.allclose(ga[~positive], gb[~positive])


def test_policy_positive_sets_and_relation_interventions():
    relations = relation_fixture()
    clip = policy_components(relations, "clip", dtype=DTYPE)
    expanded = policy_components(relations, "multipositive", dtype=DTYPE)
    grounded = policy_components(relations, "grounded", dtype=DTYPE)
    assert not clip.positives[0, 2] and clip.weights[0, 2] == 1
    assert expanded.positives[0, 2] and grounded.positives[0, 2]
    assert grounded.weights[0, 3] == 2 and grounded.weights[0, 4] == 0
    assert grounded.weights[1, 0] == 1  # unannotated cross-image pair
    assert policy_components(relations, "grounded_no_hardening").weights[0, 3] == 1
    assert policy_components(relations, "grounded_no_abstention").weights[0, 4] == 1


def test_random_exclusion_matches_per_image_count_and_preserves_positives():
    relations = torch.tensor([[1, 2, 4, 4, 3, 0, 0], [0, 0, 0, 1, 2, 4, 3]])
    a = policy_components(relations, "random_exclusion", generator=torch.Generator().manual_seed(10))
    b = policy_components(relations, "random_exclusion", generator=torch.Generator().manual_seed(10))
    assert torch.equal((a.weights == 0).sum(1), (relations == 4).sum(1))
    assert torch.equal(a.weights, b.weights)
    assert torch.all(a.weights[a.positives] == 1)
    assert torch.equal(a.positives, (relations == 1) | (relations == 2))


def test_similarity_kernel_has_exact_reconstructed_mean_cosine_definition():
    relations = torch.tensor([[1, 1, 0, 0]])
    semantic = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]], dtype=DTYPE)
    policy = policy_components(relations, "sanw_fixed", dtype=DTYPE, semantic_text_features=semantic)
    expected = torch.tensor([[1., 1., 1/(1+math.exp(5)), 1/(1+math.exp(-5))]], dtype=DTYPE)
    torch.testing.assert_close(policy.weights, expected, atol=1e-14, rtol=1e-14)
    median = policy_components(relations, "sanw_median", dtype=DTYPE, semantic_text_features=semantic)
    torch.testing.assert_close(median.weights, expected)


def test_shuffling_preserves_global_negative_weight_multiset_and_positive_set():
    relations = relation_fixture()
    _, semantic = features_fixture()
    fixed = policy_components(relations, "sanw_fixed", dtype=DTYPE, semantic_text_features=semantic)
    shuffled = policy_components(relations, "shuffled", dtype=DTYPE, semantic_text_features=semantic, generator=torch.Generator().manual_seed(93))
    duplicate = policy_components(relations, "shuffled", dtype=DTYPE, semantic_text_features=semantic, generator=torch.Generator().manual_seed(93))
    negative = ~fixed.positives
    torch.testing.assert_close(torch.sort(fixed.weights[negative]).values, torch.sort(shuffled.weights[negative]).values)
    assert torch.equal(shuffled.weights, duplicate.weights)
    assert not torch.equal(shuffled.weights[negative], fixed.weights[negative])
    assert torch.equal(fixed.positives, shuffled.positives)
    assert torch.all(shuffled.weights[shuffled.positives] == 1)


def test_smoothing_is_candidate_uniform_in_both_directions():
    relations = relation_fixture()
    policy = policy_components(relations, "smoothing", dtype=DTYPE)
    logits = torch.arange(14, dtype=DTYPE).reshape(2, 7) / 10
    expected = []
    for scores, positive in ((logits, policy.positives), (logits.T, policy.positives.T)):
        valid = positive.any(1)
        scores, positive = scores[valid], positive[valid]
        target = .9 * positive / positive.sum(1, keepdim=True) + .1 / scores.shape[1]
        expected.append(-(target * F.log_softmax(scores, 1)).sum(1).mean())
    torch.testing.assert_close(symmetric_weighted_loss(logits, policy), sum(expected) / 2)
    assert not policy.positives[0, 2]


def test_pairwise_ranking_averages_images_not_number_of_pairs():
    logits = torch.tensor([[0., 1., 2., 3.], [1., 2., 3., 4.]], dtype=DTYPE)
    relations = torch.tensor([[2, 2, 3, 3], [2, 3, 0, 0]])
    expected_first = F.softplus(torch.tensor([[2., 3.], [1., 2.]], dtype=DTYPE)).mean()
    expected_second = F.softplus(torch.tensor(1., dtype=DTYPE))
    torch.testing.assert_close(pairwise_ranking_loss(logits, relations), (expected_first + expected_second) / 2)


def test_no_eligible_ranking_image_produces_zero_gradient():
    logits = torch.tensor([[1., 2.]], dtype=DTYPE, requires_grad=True)
    loss = pairwise_ranking_loss(logits, torch.tensor([[1, 2]]))
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_backpropagate_to_adapters_but_not_semantics_or_scale(method):
    images, texts = features_fixture()
    adapter = ResidualAdapter(4).double()
    semantic = texts.clone().requires_grad_()
    scale = torch.tensor(3., dtype=DTYPE, requires_grad=True)
    adapted_images, adapted_texts = adapter(images, texts)
    loss = contrastive_loss(adapted_images, adapted_texts, relation_fixture(), method, scale,
                            generator=torch.Generator().manual_seed(9), semantic_text_features=semantic)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert adapter.image.weight.grad is not None and torch.isfinite(adapter.image.weight.grad).all()
    assert adapter.text.weight.grad is not None and torch.isfinite(adapter.text.weight.grad).all()
    assert adapter.image.weight.grad.norm() > 0 and adapter.text.weight.grad.norm() > 0
    assert semantic.grad is None and scale.grad is None


def test_adapter_zero_initialization_identity_and_parameter_count():
    adapter = ResidualAdapter()
    assert sum(p.numel() for p in adapter.parameters()) == 524288
    assert all(torch.count_nonzero(p) == 0 for p in adapter.parameters())
    x = F.normalize(torch.randn(3, 512), dim=1)
    torch.testing.assert_close(adapter.encode_image(x), x)
    torch.testing.assert_close(adapter.encode_text(x), x)
    with torch.no_grad():
        adapter.image.weight[0, 0] = .5
    torch.testing.assert_close(adapter.encode_image(x).norm(dim=1), torch.ones(3))
    torch.testing.assert_close(adapter.encode_text(x), x)


def test_invalid_inputs_fail_clearly():
    with pytest.raises(ValueError, match="requires frozen"):
        policy_components(relation_fixture(), "sanw_fixed")
    with pytest.raises(ValueError, match="Unknown method"):
        policy_components(relation_fixture(), "not_a_method")
    with pytest.raises(ValueError, match="one on positives"):
        weighted_row_loss(torch.zeros(1, 2), torch.tensor([[True, False]]), torch.tensor([[0., 1.]]))
    with pytest.raises(ValueError, match="unit candidate weights"):
        weighted_row_loss(torch.zeros(1, 2), torch.tensor([[True, False]]), torch.tensor([[1., .5]]), .1)


def test_median_threshold_uses_only_negative_cells_and_actual_batch_median():
    relations = torch.tensor([[1, 0, 0, 0]])
    semantic = torch.tensor([[1., 0.], [.8, .6], [.6, .8], [-.6, .8]], dtype=DTYPE)
    policy = policy_components(relations, "sanw_median", dtype=DTYPE, semantic_text_features=semantic)
    expected = torch.tensor([[1., 1/(1+math.exp(2)), .5, 1/(1+math.exp(-12))]], dtype=DTYPE)
    torch.testing.assert_close(policy.weights, expected, atol=1e-13, rtol=1e-13)


def test_empty_positive_rows_are_skipped_in_mean_and_receive_zero_gradient():
    logits = torch.tensor([[1., 3.], [2., 4.]], dtype=DTYPE, requires_grad=True)
    positives = torch.tensor([[True, False], [False, False]])
    actual = weighted_row_loss(logits, positives, torch.ones_like(logits))
    expected = F.softplus(torch.tensor(2., dtype=DTYPE))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2, dtype=DTYPE))


def test_policy_weights_are_detached_even_when_semantic_features_require_grad():
    relations = relation_fixture()
    _, semantic = features_fixture()
    semantic.requires_grad_()
    for method in ("sanw_fixed", "sanw_median", "shuffled"):
        policy = policy_components(relations, method, dtype=DTYPE, semantic_text_features=semantic)
        assert not policy.weights.requires_grad
        assert not policy.positives.requires_grad


def test_seeded_random_exclusion_is_a_random_assignment_not_neutral_mask():
    relations = torch.tensor([[1, 2, 4, 4, 3, 0, 0]])
    draws = [policy_components(relations, "random_exclusion", generator=torch.Generator().manual_seed(seed)).weights == 0
             for seed in range(10)]
    assert any(not torch.equal(mask, relations == 4) for mask in draws)
    assert len({tuple(mask.flatten().tolist()) for mask in draws}) > 1
