"""Focused checks of the new development statistic and bootstrap distribution."""
from itertools import product
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from gcr.practical_development_v7 import (
    calibrated_score, development_bootstrap, grouped_bootstrap_moments,
    select_calibrated_development, source_pair_validation_metrics,
)


def test_grouped_bootstrap_distribution_moments_match_exact_cluster_draws():
    # Four equal-size clusters, with repeated sums: multinomial grouping must
    # preserve their frequency, not assign equal probability to distinct sums.
    delta = np.array([-1, 0, -1, 0, 0, 1, 1, 1])
    clusters = np.repeat(np.arange(4), 2)
    sums = delta.reshape(4, 2).sum(axis=1)
    exact = np.array([sum(sums[list(draw)]) / 8 for draw in product(range(4), repeat=4)])
    analytic = grouped_bootstrap_moments(delta, clusters)
    assert analytic["mean"] == pytest.approx(exact.mean(), abs=1e-15)
    assert analytic["variance"] == pytest.approx(exact.var(), abs=1e-15)
    summary, samples = development_bootstrap(delta, clusters, replicates=100000)
    assert summary["algorithm"] == "grouped_multinomial_equal_image_cluster_size"
    assert abs(samples.mean() - exact.mean()) < 0.003
    assert abs(samples.var() - exact.var()) < 0.001
    assert set(np.unique(samples)).issubset(set(np.unique(exact)))
    # Independent direct ID resampling has the same first two distribution moments.
    direct = sums[np.random.default_rng(20261007).integers(0, 4, size=(100000, 4))].sum(axis=1) / 8
    assert abs(samples.mean() - direct.mean()) < 0.004
    assert abs(samples.var() - direct.var()) < 0.001


def test_unequal_cluster_sizes_fall_back_to_original_ratio_bootstrap():
    delta = np.array([-1, 1, 0, 1, 1])
    clusters = np.array([0, 1, 1, 2, 2])
    summary, actual = development_bootstrap(delta, clusters, replicates=777)
    rng = np.random.default_rng(20261007)
    sums, sizes, expected = np.array([-1, 1, 2]), np.array([1, 2, 2]), []
    for start in range(0, 777, 250):
        draws = rng.integers(0, 3, size=(min(250, 777 - start), 3))
        expected.extend(sums[draws].sum(axis=1) / sizes[draws].sum(axis=1))
    assert summary["algorithm"] == "direct_paired_image_cluster_draws_unequal_sizes"
    assert np.array_equal(actual, expected)
    with pytest.raises(ValueError):
        grouped_bootstrap_moments(delta, clusters)


def test_source_validation_has_no_training_margin_filter_and_is_image_balanced():
    # Image0 has 3 distinct source pairs x2 negatives. Image1 has one pair x1
    # negative. Easy +2 and impossible -2 margins both remain in validation.
    texts = ["a", "b", "c", "no a", "no b", "d", "e", "no d", " D! "]
    data = SimpleNamespace(split_indices={"validation": [0, 1]}, image_ids=["a", "b"],
        pairs={0: {0: 1, 1: 1, 2: 1, 3: 3, 4: 3}, 1: {5: 1, 6: 1, 7: 3, 8: 3}},
        manifest={"texts": [{"text": text} for text in texts]})
    images = np.array([0]*5+[1]*4)
    values = np.array([2, 2, 2, 0, 0, -2, -2, 0, 0])
    summary, raw = source_pair_validation_metrics(data, images, np.arange(9), values)
    assert summary["triplet_count"] == 7
    assert summary["paired_joint_accuracy"] == .5  # image balanced, not 6/7
    assert summary["excluded_conflicting_negative_count"] == 1
    assert not summary["frozen_margin_filter"]
    assert np.array_equal(raw["triplet_count"], [6, 1])
    # Strict both-positive rule: equality is failure.
    values[:3] = 0
    assert source_pair_validation_metrics(data, images, np.arange(9), values)[0]["paired_joint_accuracy"] == 0


def test_global_alpha_score_and_uncertainty_selector_boundaries():
    base, correction = np.array([.2, .3]), np.array([.01, -.01])
    assert np.array_equal(calibrated_score(base, correction, .5), base + .5 * correction)
    with pytest.raises(ValueError):
        calibrated_score(base, correction, 0)
    effect = {"difference": 0., "ci_lower": -.009, "replicates": 100000,
              "family_size": 80, "bootstrap_seed": 20261007}
    row = {"epoch": 2, "alpha": .5, "optimizer_steps": 4, "update_norm": .1,
           "residual_rms": .001, "checkpoint": {"path": "example.pt", "sha256": "fixture"},
           "composition": {"paired_joint_accuracy": .6, "mean_paired_joint_margin": .1},
           "frozen_composition": {"paired_joint_accuracy": .5},
           "retention": {"i2t": copy.deepcopy(effect), "t2i": copy.deepcopy(effect)}}
    other = copy.deepcopy(row); other["alpha"] = .2
    assert select_calibrated_development([row, other])["selected_alpha"] == .2
    bad = copy.deepcopy(row); bad["retention"]["t2i"]["ci_lower"] = -.01
    assert select_calibrated_development([bad])["selected_epoch"] is None
    bad["retention"]["t2i"]["ci_lower"] = -.009
    bad["retention"]["i2t"]["difference"] = -1e-12
    assert select_calibrated_development([bad])["selected_epoch"] is None
    bad["retention"] = copy.deepcopy(row["retention"]); bad["update_norm"] = 0
    assert select_calibrated_development([bad])["selected_epoch"] is None
    bad = copy.deepcopy(row); bad["composition"]["paired_joint_accuracy"] = .5
    assert select_calibrated_development([bad])["selected_epoch"] is None
