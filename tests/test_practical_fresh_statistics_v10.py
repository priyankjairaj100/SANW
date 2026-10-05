"""Exact synthetic reference checks, including roundoff-screen ties and strict gates."""
from fractions import Fraction

import numpy as np
import pytest

from gcr.practical_fresh_statistics_v10 import (
    TAIL, _bootstrap_core, _candidate_sets, _roundoff_envelope,
    _validated_counts, exact_lower_exceeds, exact_mean_nonnegative, image_balanced_bootstrap,
)


def brute(n, d, replicates, seed):
    ratios = [Fraction(int(a), int(b)) for a, b in zip(n, d)]
    draws = np.random.default_rng(seed).integers(0, len(n), size=(replicates, len(n)))
    exact = [sum((ratios[int(i)] for i in row), Fraction(0)) / len(n) for row in draws]
    ordered = sorted(exact)
    endpoints = []
    for probability in (TAIL, 1 - TAIL):
        position = probability * (replicates - 1)
        lower = position.numerator // position.denominator
        weight = position - lower
        endpoints.append((1-weight)*ordered[lower] + weight*ordered[min(lower+1, replicates-1)])
    return sum(ratios, Fraction(0))/len(n), endpoints, exact


@pytest.mark.parametrize("force", [False, True])
def test_variable_denominator_image_mean_and_percentiles_match_fraction_reference(force):
    # Unequal triplet counts: pooled numerator/denominator is the wrong estimand.
    n, d = np.array([1, -2, 5, -3, 0]), np.array([3, 6, 21, 9, 30])
    expected_mean, endpoints, exact_draws = brute(n, d, 911, 1234)
    mean, low, high, samples, metadata = _bootstrap_core(n, d, replicates=911, seed=1234, force_fallback=force)
    assert mean == expected_mean != Fraction(int(n.sum()), int(d.sum()))
    assert [low, high] == endpoints
    assert len(samples) == len(exact_draws)
    if force:
        bound = metadata["floating_screen_error_bound"]
        assert all(abs(Fraction.from_float(float(a))-b) <= Fraction.from_float(bound) for a, b in zip(samples, exact_draws))
        assert metadata["exact_fallback_draws"] > 0
    else:
        assert samples.tolist() == [float(x) for x in exact_draws]
        assert metadata["exact_fallback_draws"] == 0


def test_large_common_denominator_automatically_uses_exact_fallback():
    n = np.array([1, -1, 2], dtype=np.int64)
    d = np.array([3*10000000019, 3*10000000033, 3*10000000061], dtype=np.int64)
    mean, low, high, _, metadata = _bootstrap_core(n, d, replicates=533, seed=190)
    expected_mean, endpoints, _ = brute(n, d, 533, 190)
    assert mean == expected_mean and [low, high] == endpoints
    assert metadata["common_denominator_bits"] > 63
    assert metadata["path"].startswith("certified_float_screen")


def test_screen_includes_all_ties_and_excludes_only_provably_outside_values():
    lower = np.array([-4., -1., 0., 0., 0., 2., 7.])
    upper = np.array([-3., -.5, 0., 0., 1., 3., 9.])
    candidates = _candidate_sets(lower, upper, {2, 3, 4})
    for rank in (2, 3):
        indices, local = candidates[rank]
        assert indices.tolist() == [2, 3, 4]
        assert local == rank - 2
    indices, local = candidates[4]
    assert indices.tolist() == [2, 3, 4] and local == 2


def test_forced_screen_with_many_equal_draws_retains_exact_zero_boundaries():
    n, d = np.array([1, -2, 0, 0]), np.array([3, 6, 3, 9])
    expected_mean, endpoints, _ = brute(n, d, 701, 37)
    mean, low, high, _, metadata = _bootstrap_core(n, d, replicates=701, seed=37, force_fallback=True)
    assert mean == expected_mean == 0 and [low, high] == endpoints
    assert metadata["exact_fallback_draws"] > 4


def test_distinct_exact_values_that_all_round_to_same_float_are_sorted_exactly():
    first = 3*((1 << 61)-1)
    d = np.array([first, first-6], dtype=np.int64)
    n = np.array([first//2, (first-6)//2+1], dtype=np.int64)
    mean, low, high, samples, metadata = _bootstrap_core(n, d, replicates=301, seed=82)
    expected_mean, endpoints, _ = brute(n, d, 301, 82)
    assert mean == expected_mean and [low, high] == endpoints
    assert np.all(samples == .5)
    assert low < Fraction(1, 2) < high
    assert metadata["exact_fallback_draws"] == 301


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("n,d,boundary", [([0, 1], [3, 3], Fraction(0)), ([-3, -6], [300, 600], Fraction(-1, 100))])
def test_actual_bootstrap_lower_endpoint_equality_fails_strict_gate(force, n, d, boundary):
    _, low, _, _, _ = _bootstrap_core(np.array(n), np.array(d), replicates=1051, seed=91, force_fallback=force)
    assert low == boundary
    effect = {"ci_lower": float(low), "exact_ci_lower": {"numerator": low.numerator, "denominator": low.denominator}}
    assert exact_lower_exceeds(effect, boundary) is False


def test_error_envelope_covers_adversarial_cancellation_and_conversion():
    ratios = [Fraction(1, 3), Fraction(-1, 3), Fraction(1, 3*((1 << 61)-1)), Fraction(1), Fraction(-1)]
    values, exact_bound, float_bound = _roundoff_envelope(ratios)
    assert Fraction.from_float(float_bound) >= exact_bound
    for indices in ([0, 1, 2, 3, 4], [3, 0, 2, 1, 4], [2]*5, [0, 3, 4, 4, 1]):
        observed = float(values[indices].sum(dtype=np.float64)/len(indices))
        exact = sum((ratios[i] for i in indices), Fraction(0))/len(indices)
        assert abs(Fraction.from_float(observed)-exact) <= exact_bound


def test_exact_gate_equality_and_negative_zero_do_not_pass_strict_thresholds():
    for value, threshold in ((Fraction(0), 0), (Fraction(-1, 100), Fraction(-1, 100))):
        effect = {"ci_lower": float(value), "exact_ci_lower": {"numerator": value.numerator, "denominator": value.denominator}}
        assert exact_lower_exceeds(effect, threshold) is False
        positive = value + Fraction(1, 10**30)
        effect = {"ci_lower": float(positive), "exact_ci_lower": {"numerator": positive.numerator, "denominator": positive.denominator}}
        assert exact_lower_exceeds(effect, threshold) is True
    with pytest.raises(TypeError):
        exact_lower_exceeds(effect, -.01)
    effect = {"difference": -0.0, "exact_difference": {"numerator": 0, "denominator": 3}}
    assert exact_mean_nonnegative(effect)
    effect["difference"] = .1
    with pytest.raises(ValueError, match="Displayed"):
        exact_mean_nonnegative(effect)


def test_native_bounds_validated_before_conversion_and_invalid_denominators_rejected():
    for n, d in ((np.array([1], dtype=np.uint64), np.array([2**63], dtype=np.uint64)),
                 (np.array([2**63], dtype=np.uint64), np.array([3], dtype=np.uint64)),
                 (np.array([np.iinfo(np.int64).min]), np.array([3])),
                 (np.array([1]), np.array([2])), (np.array([4]), np.array([3])),
                 (np.array([1.]), np.array([3]))):
        with pytest.raises(ValueError):
            _validated_counts(n, d)


def test_public_fixed_contract_permutation_invariance_and_duplicate_cluster_rejection():
    n, d = np.array([1, -1]), np.array([3, 6])
    effect, samples = image_balanced_bootstrap(n, d, np.array(["z", "a"]))
    alternate, other_samples = image_balanced_bootstrap(n[::-1], d[::-1], np.array(["a", "z"]))
    assert effect == alternate
    np.testing.assert_array_equal(samples, other_samples)
    assert effect["replicates"] == 100000 and effect["bootstrap_seed"] == 20261007
    assert effect["family_size"] == 80 and effect["fixed_training_seeds"] == [17, 29, 43]
    assert effect["exact_difference"] == {"numerator": 1, "denominator": 12}
    with pytest.raises(ValueError, match="distinct"):
        image_balanced_bootstrap(n, d, ["a", "a"])
