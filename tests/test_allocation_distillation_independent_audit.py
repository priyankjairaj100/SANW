"""Known examples for the independent audit, without production statistics."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_allocation_distillation_independent import (
    BOOTSTRAP_SEED, SEEDS, bootstrap_matrix, gate_checks, reconstruct_metrics,
    strategy_arrays, summarize_effect,
)


def test_all_100000_draws_match_direct_item_resampling_with_unequal_clusters():
    # The two clusters contain two items and one item. Equal cluster averaging is wrong.
    clusters = np.array(["b", "a", "a"])
    delta = np.tile(np.array([-1., 1., 0.]), (3, 1))
    samples = bootstrap_matrix([delta], clusters)[:, 0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected = rng.integers(0, 2, size=(100000, 2))
    # All four ordered cluster resamples have a hand-computed outcome.
    known = {(0, 0): .5, (0, 1): 0., (1, 0): 0., (1, 1): -1.}
    expected = np.asarray([known[tuple(draw)] for draw in selected])
    assert np.array_equal(samples, expected)
    for family_size in (80, 24):
        result = summarize_effect(delta, samples, clusters, family_size)
        assert result["difference"] == 0.
        assert result["ci_lower"] == -1.
        assert result["ci_upper"] == .5
        assert result["items"] == 3 and result["images"] == 2


def test_batch_size_does_not_change_the_frozen_draw_stream():
    values = np.array([[[0., 1., .3, -.8], [1., 0., .7, -.4], [.2, .8, .5, -.6]]])
    clusters = np.array(["z", "a", "a", "m"])
    a = bootstrap_matrix(values, clusters, replicates=2003, batch_size=17)
    b = bootstrap_matrix(values, clusters, replicates=2003, batch_size=250)
    assert np.allclose(a, b, rtol=0, atol=2e-16)


def test_linear_quantile_interpolation_and_fixed_seed_variation():
    delta = np.array([[1., -1.], [0., 0.], [-.5, .5]])
    samples = np.arange(10, dtype=float)
    result = summarize_effect(delta, samples, ["a", "b"], 1)
    assert result["ci_lower"] == pytest.approx(.225)
    assert result["ci_upper"] == pytest.approx(8.775)
    assert result["seed_differences"] == [0., 0., 0.]
    assert result["seed_difference_std"] == 0.


def test_random_draws_average_equally_within_each_fixed_seed():
    runs, predictions = [], {}
    for seed_index, seed in enumerate(SEEDS):
        for draw in range(3):
            sid = f"{seed}-{draw}"
            runs.append({"state_id": sid, "seed": seed, "draw_id": draw, "method": "matched"})
            predictions[(sid, "sugarcrepe_pp")] = {
                "item_ids": np.array(["1", "2", "3"]), "image_ids": np.array(["a", "a", "b"]),
                "correct": np.array([float(seed_index == 0), float(draw == 0), float(draw == seed_index)])}
    result, _, clusters = strategy_arrays(runs, predictions, "sugarcrepe_pp", "both_accuracy")
    assert np.allclose(result, [[1., 1/3, 1/3], [0., 1/3, 1/3], [0., 1/3, 1/3]])
    assert clusters.tolist() == ["a", "a", "b"]
    with pytest.raises(ValueError, match="complete draws"):
        strategy_arrays(runs[:-1], predictions, "sugarcrepe_pp", "both_accuracy")


def test_relation_scores_receive_half_credit_for_exact_ties():
    raw = {"image_ids": np.array(["a", "b"]), "image_accuracy": np.array([.5, 1.]),
        "comparison_counts": np.array([2, 1]), "pair_image_ids": np.array(["a", "a", "a", "b", "b"]),
        "pair_text_ids": np.array(["1", "2", "3", "4", "5"]),
        "pair_relations": np.array(["supported", "supported", "contradicted", "supported", "contradicted"]),
        "pair_scores": np.array([.5, .5, .5, 1., 0.])}
    metrics = reconstruct_metrics("visual_entailment", raw)
    assert metrics["accuracy"] == .75
    assert metrics["comparison_count"] == 3


def test_both_positive_triplets_use_strict_comparisons():
    raw = {"image_ids": np.array(["a", "a", "b"]), "positive1_scores": np.array([1., 0., 2.]),
        "positive2_scores": np.array([0., 2., 3.]), "negative_scores": np.array([0., 0., 1.]),
        "positive1_correct": np.array([True, False, True]), "positive2_correct": np.array([False, True, True]),
        "correct": np.array([False, False, True])}
    metrics = reconstruct_metrics("sugarcrepe_pp", raw)
    assert metrics["accuracy"] == metrics["both_accuracy"] == 1/3
    assert metrics["positive1_accuracy"] == metrics["positive2_accuracy"] == 2/3


def test_practical_gate_rejects_exact_boundaries_and_zero_epoch():
    runs = [{"seed": seed, "epoch": 1, "update_norm": 1.} for seed in SEEDS]
    rows = [{"dataset": "e_vil_test1000", "metric": "i2t.r1", "ci_lower": -.01},
            {"dataset": "e_vil_test1000", "metric": "t2i.r1", "ci_lower": -.009},
            {"dataset": "sugarcrepe_pp", "metric": "both_accuracy", "ci_lower": 0.}]
    checks = gate_checks(runs, rows)
    assert checks == {"all_three_nonzero": True, "i2t_retention": False,
                      "t2i_retention": True, "sugarcrepe_pp_improvement": False}
    rows[0]["ci_lower"] = -.009
    rows[-1]["ci_lower"] = .001
    assert all(gate_checks(runs, rows).values())
    runs[0]["epoch"] = 0
    assert not gate_checks(runs, rows)["all_three_nonzero"]
