import copy

import numpy as np
import pytest

from gcr.practical_inner_evaluation_v9 import (
    LABCLIPScorer, exact_changes, paired_changes, select_shared,
)


def metric_arrays(original):
    result = {}
    for name in ("i2t", "t2i", "original", "source_pair"):
        result[f"{name}_cluster_ids"] = np.array(["a", "b"])
        result[f"{name}_correct"] = np.array(original if name == "original" else [0., 1.])
        if name in ("original", "source_pair"):
            result[f"{name}_triplet_counts"] = np.array([3, 3])
    return result


def test_exact_counts_remove_false_gain_from_ratio_cancellation():
    frozen, trained = metric_arrays([1 / 3, 2 / 3]), metric_arrays([0., 1.])
    assert (trained["original_correct"] - frozen["original_correct"]).mean() != 0
    changes, raw = paired_changes(frozen, trained)
    assert changes["original"] == 0
    assert exact_changes(raw)["original"] == {"numerator": 0, "denominator": 1}


def test_noninteger_accuracy_and_changed_units_rejected():
    frozen, trained = metric_arrays([0., 1.]), metric_arrays([.2, 1.])
    with pytest.raises(ValueError, match="reconstructible"):
        paired_changes(frozen, trained)
    trained = metric_arrays([0., 1.])
    trained["original_cluster_ids"] = np.array(["b", "a"])
    with pytest.raises(ValueError, match="units"):
        paired_changes(frozen, trained)


def grid():
    result = []
    for encoder in ("vit_b32", "rn50"):
        for radius in (.1, .3, 1.):
            for weight in (.25, 1., 4.):
                result.append({"encoder": encoder, "radius": radius, "composition_weight": weight,
                               "nonzero_functional_update": True,
                               "exact_paired_changes": {name: {"numerator": 1 if name == "original" else 0,
                                                               "denominator": 100}
                                                        for name in ("i2t", "t2i", "original", "source_pair")}})
    return result


def test_complete_shared_grid_and_predeclared_ties():
    rows = grid()
    assert select_shared(rows, family="joint")["selected_key"] == [.1, 1.]
    for row in rows:
        if row["composition_weight"] == 1.:
            row["exact_paired_changes"]["original"]["numerator"] = 0
    assert select_shared(rows, family="joint")["selected_key"] == [.1, .25]
    with pytest.raises(ValueError, match="coverage"):
        select_shared(rows[:-1], family="joint")
    with pytest.raises(ValueError, match="duplicate"):
        select_shared(rows + [rows[0]], family="joint")


def test_minimax_precedes_sum_and_one_encoder_failure_rejects_shared_cell():
    rows = grid()
    for row in rows:
        if row["radius"] == .3 and row["composition_weight"] == 1:
            row["exact_paired_changes"]["original"]["numerator"] = 2
        if row["radius"] == 1. and row["composition_weight"] == 1:
            row["exact_paired_changes"]["original"]["numerator"] = 100 if row["encoder"] == "rn50" else 1
    assert select_shared(rows, family="joint")["selected_key"] == [.3, 1.]
    for row in rows:
        if row["radius"] == .3 and row["encoder"] == "rn50":
            row["exact_paired_changes"]["i2t"]["numerator"] = -1
    assert select_shared(rows, family="joint")["selected_key"] == [1., 1.]


def test_labclip_requires_learned_gain_beyond_identity_and_prefers_early_epoch():
    rows = []
    for encoder in ("vit_b32", "rn50"):
        for epoch in (1, 2, 4, 8, 16, 32):
            changes = {name: {"numerator": 1 if name == "original" else 0, "denominator": 100}
                       for name in ("i2t", "t2i", "original", "source_pair")}
            identity = copy.deepcopy(changes)
            identity["original"]["numerator"] = 0 if epoch == 1 else 1
            rows.append({"encoder": encoder, "epoch": epoch, "nonzero_functional_update": True,
                         "exact_paired_changes": changes, "exact_identity_paired_changes": identity})
    assert select_shared(rows, family="labclip")["selected_key"] == [2]


def test_labclip_canonical_transform_matches_official_adapter_and_batching():
    from gcr.practical_labclip_v9 import canonical_transform
    rng = np.random.default_rng(88)
    x, t, weight = rng.normal(size=(9, 7)), rng.normal(size=(9, 7)), np.eye(7) + .1 * rng.normal(size=(7, 7))
    scorer = LABCLIPScorer(weight)
    assert np.array_equal(scorer.transform(t), canonical_transform(t, weight))
    assert np.array_equal(scorer.pair_scores(x, t), np.array([scorer.pair_scores(x[i:i+1], t[i:i+1])[0] for i in range(9)]))
