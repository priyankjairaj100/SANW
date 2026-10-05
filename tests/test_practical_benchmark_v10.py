"""Toy-only tests: no benchmark feature bank, labels, or outcomes are opened."""
from pathlib import Path
from fractions import Fraction
import sys
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gcr.practical_benchmark_v10 import (bootstrap, practical_gate, score_triplets, score_retrieval,
                                       paired_seed_differences, SEEDS)
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer
from evaluate_practical_v6 import bootstrap as recovered_bootstrap
import evaluate_practical_benchmark_v10 as cli
from prepare_practical_benchmark_inputs_v10 import inventory, required_paths
from gcr.practical_exact_bootstrap_v10 import integer_cluster_bootstrap, lower_exceeds, exact_percentile


def test_bootstrap_matches_recovered_draws_with_unequal_image_cluster_sizes():
    # One image has three triplets and one has one: equal-image averaging would
    # incorrectly report zero instead of the inherited item mean of one half.
    delta = np.array([[1, 1, 1, -1]] * 3, dtype=np.int64)
    clusters = np.array(["a", "a", "a", "b"])
    current, actual = bootstrap(delta, clusters)
    old, expected = recovered_bootstrap(delta.astype(float), clusters)
    np.testing.assert_array_equal(actual, expected)
    assert current["difference"] == .5
    assert current["ci_lower"] == old["ci_lower"] and current["ci_upper"] == old["ci_upper"]
    assert current["images"] == 2 and current["items"] == 4
    assert current["replicates"] == 100000 and current["family_size"] == 80 and current["bootstrap_seed"] == 20261007
    assert current["tail_probability"] == .0003125


def test_seed_mean_is_fixed_and_exact_cancellation_remains_zero():
    delta = np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]], dtype=np.int64)
    effect, draws = bootstrap(delta, np.array(["a", "b", "c"]))
    assert effect["exact_difference"] == {"numerator": 0, "denominator": 1}
    assert np.all(draws == 0)
    with pytest.raises(ValueError):
        bootstrap(delta[:2], np.array(["a", "b", "c"]))


def dataset():
    arrays = {"image_features": np.array([[1., 0.], [0., 1.]], dtype=np.float32),
              "text_features": np.array([[1., 0.], [0., 1.], [.8, .6]], dtype=np.float32),
              "image_ids": np.array(["a", "b"]), "text_ids": np.array(["p1", "p2", "n"])}
    rows = [{"id": "item", "image_id": "a", "category": "toy", "positive1_id": "p1", "positive2_id": "p2", "negative_id": "n"}]
    return {"arrays": arrays, "manifest": {"triplets": rows, "pairs": [
        {"image_id": "a", "text_id": "p1", "relation": "source"},
        {"image_id": "b", "text_id": "p2", "relation": "source"},
        {"image_id": "a", "text_id": "n", "relation": "source"}]}}


def test_both_positive_endpoint_requires_both_and_strict_ties_fail():
    data = dataset()
    _, raw = score_triplets(data)
    assert raw["positive1_correct"].tolist() == [True] and raw["correct"].tolist() == [False]
    data["manifest"]["triplets"][0]["positive2_id"] = "p1"
    assert score_triplets(data)[1]["correct"].tolist() == [True]
    data["manifest"]["triplets"][0]["negative_id"] = "p1"
    assert score_triplets(data)[1]["correct"].tolist() == [False]


def test_same_canonical_score_is_used_for_pairs_and_full_gallery_ties():
    data = dataset()
    data["arrays"]["text_features"][2] = data["arrays"]["text_features"][0]
    scorer = CanonicalScorer(np.zeros(2), np.zeros(2), np.eye(2), np.eye(2), .1 * np.eye(2))
    _, retrieval = score_retrieval(data, scorer)
    _, triplet = score_triplets(data, scorer)
    assert retrieval["i2t_top_indices"].tolist() == [0, 1]
    assert retrieval["i2t_top_scores"][0] == triplet["positive1_scores"][0] == triplet["negative_scores"][0]


def test_pairing_rejects_changed_gallery_or_seed_coverage():
    _, raw = score_retrieval(dataset())
    selected = {seed: {k: v.copy() for k, v in raw.items()} for seed in SEEDS}
    reference = {seed: raw for seed in SEEDS}
    delta, clusters = paired_seed_differences("e_vil_test1000", "t2i.r1", selected, reference)
    assert delta.shape == (3, 3) and np.array_equal(clusters, raw["text_source_image_ids"])
    selected[29]["text_ids"][0] = "bad"
    with pytest.raises(ValueError, match="identities"):
        paired_seed_differences("e_vil_test1000", "t2i.r1", selected, reference)
    with pytest.raises(ValueError, match="fixed seeds"):
        paired_seed_differences("e_vil_test1000", "t2i.r1", {17: raw}, reference)


def effects():
    return [{"encoder": encoder, "dataset": name, "metric": metric, "contrast": "joint_minus_frozen", "ci_lower": .001,
             "exact_ci_lower": {"numerator": 1, "denominator": 1000}}
            for encoder in ("vit_b32", "rn50") for name, metric in (("e_vil_test1000", "i2t.r1"), ("e_vil_test1000", "t2i.r1"),
             ("sugarcrepe_pp", "both_accuracy"), ("coco_karpathy", "i2t.r1"), ("coco_karpathy", "t2i.r1"))]


@pytest.mark.parametrize("dataset_name,boundary", [("e_vil_test1000", -.01), ("sugarcrepe_pp", 0.)])
def test_primary_threshold_equality_fails(dataset_name, boundary):
    rows = effects()
    next(row for row in rows if row["dataset"] == dataset_name)["ci_lower"] = boundary
    next(row for row in rows if row["dataset"] == dataset_name)["exact_ci_lower"] = (
        {"numerator": -1, "denominator": 100} if boundary else {"numerator": 0, "denominator": 1})
    assert practical_gate(rows)["passed"] is False


def test_coco_is_secondary_and_controls_cannot_replace_failed_candidate():
    rows = effects()
    for row in rows:
        if row["dataset"] == "coco_karpathy":
            row["ci_lower"] = -1
            row["exact_ci_lower"] = {"numerator": -1, "denominator": 1}
    assert practical_gate(rows)["passed"] is True
    assert practical_gate(rows)["encoders"]["vit_b32"]["secondary_coco_retention"] is False
    rows[0]["ci_lower"] = -.02
    rows[0]["exact_ci_lower"] = {"numerator": -1, "denominator": 50}
    rows += [{**row, "contrast": "no_retention_minus_frozen", "ci_lower": .5} for row in effects()]
    assert practical_gate(rows)["passed"] is False


def test_missing_or_failed_development_gate_blocks_before_any_feature_load():
    with patch.object(cli, "read", return_value={"study": "sanw_practical_v10_fixed_seed_development_gate", "passed": False}), \
         patch.object(cli, "load_dataset") as loader:
        with pytest.raises(ValueError, match="must pass"):
            cli.verify_development_gate("fake", "protocol")
        loader.assert_not_called()


def test_recovery_inventory_only_names_prespecified_files():
    plan = inventory()
    names = {value for by_dataset in required_paths().values() for entry in by_dataset.values() for value in entry.values()}
    assert len(names) == 15 and plan["required_unique_file_count"] == 15
    assert all("/sugarcrepe/" not in name for name in names)
    assert plan["no_model_scoring"] is True


def test_zero_sc_gain_cannot_pass_from_rounded_thirds():
    delta = np.zeros((3, 6), dtype=np.int64)
    delta[0] = [1, 1, 1, -1, -1, -1]
    clusters = np.zeros(6, dtype=np.int64)
    legacy, _ = recovered_bootstrap(delta, clusters)
    assert legacy["ci_lower"] > 0  # The concrete inherited cancellation bug.
    effect, samples = bootstrap(delta, clusters)
    assert np.all(samples == 0) and effect["exact_ci_lower"] == {"numerator": 0, "denominator": 1}
    assert not lower_exceeds(effect, Fraction(0))


def test_three_seed_development_exact_minus_one_percent_must_fail():
    numerator = np.zeros(900, dtype=np.int64)
    numerator[:39], numerator[39:75] = 1, -1
    effect, _ = integer_cluster_bootstrap(numerator, 3, np.arange(900), grouped=True)
    assert effect["difference"] == 1 / 900
    assert effect["exact_ci_lower"] == {"numerator": -1, "denominator": 100}
    assert effect["ci_lower"] == -.01 and not lower_exceeds(effect, Fraction(-1, 100))
    with pytest.raises(TypeError):
        lower_exceeds(effect, -.01)


def test_percentile_interpolation_uses_exact_probability_and_variable_denominators():
    n, d = np.array([0, 1, 1, 1]), np.array([1, 7, 3, 2])
    assert exact_percentile(n, d, Fraction(1, 3200)) == Fraction(3, 22400)
    assert exact_percentile(n, d, Fraction(1, 2)) == (Fraction(1, 7) + Fraction(1, 3)) / 2
