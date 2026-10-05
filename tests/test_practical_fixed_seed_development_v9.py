from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aggregate_practical_official_development_v9 import aggregate_gate, fixed_seed_pairs
from gcr.practical_inner_evaluation_v9 import METRICS, exact_changes, paired_changes, rational


def outcomes(original, retrieval=(0, 1)):
    result = {}
    for metric in METRICS:
        result[f"{metric}_cluster_ids"] = np.array(["a", "b"])
        result[f"{metric}_correct"] = np.asarray(retrieval if metric in ("i2t", "t2i") else original, dtype=float)
        if metric in ("original", "source_pair"):
            result[f"{metric}_triplet_counts"] = np.array([10, 10])
    return result


def pairs():
    baseline = outcomes([.5, .5])
    return {17: paired_changes(baseline, outcomes([.6, .6], (1, 1)))[1],
            29: paired_changes(baseline, outcomes([.4, .4], (0, 0)))[1],
            43: paired_changes(baseline, outcomes([.6, .6], (1, 1)))[1]}


def test_fixed_seed_mean_is_exact_and_keeps_a_failing_replication():
    combined = fixed_seed_pairs(pairs())
    assert combined["original_difference_numerators"].tolist() == [1, 1]
    assert combined["original_difference_denominators"].tolist() == [30, 30]
    exact = exact_changes(combined)
    assert str(rational(exact["original"])) == "1/30"
    assert str(rational(exact["i2t"])) == "1/6"
    assert np.array_equal(combined["i2t_difference"], np.array([2 / 3, -1 / 3]))
    effects = {metric: {"ci_lower": 0.} for metric in METRICS}
    gate = aggregate_gate(exact, effects, [.1, .1, .1])
    assert gate["passed"]
    assert gate["individual_replication_pass_required"] is False
    assert gate["seed_resampling"] is False


def test_fixed_seed_coverage_and_cluster_alignment_are_mandatory():
    values = pairs()
    with pytest.raises(ValueError, match="exactly"):
        fixed_seed_pairs({17: values[17], 29: values[29]})
    values[29]["i2t_cluster_ids"] = np.array(["b", "a"])
    with pytest.raises(ValueError, match="identical paired queries"):
        fixed_seed_pairs(values)


def test_fixed_seed_gate_preserves_strict_minus_one_pp_and_nonzero_requirements():
    exact = exact_changes(fixed_seed_pairs(pairs()))
    effects = {metric: {"ci_lower": 0.} for metric in METRICS}
    effects["i2t"]["ci_lower"] = -.01
    assert not aggregate_gate(exact, effects, [.1, .1, .1])["passed"]
    effects["i2t"]["ci_lower"] = np.nextafter(-.01, np.inf)
    assert aggregate_gate(exact, effects, [.1, .1, .1])["passed"]
    with pytest.raises(ValueError, match="nonzero"):
        aggregate_gate(exact, effects, [.1, 0., .1])
