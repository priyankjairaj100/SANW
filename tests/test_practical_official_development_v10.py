"""Synthetic validation of unchanged gates and fixed-three-seed v10 arithmetic."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_practical_official_development_v10 as official
import aggregate_practical_official_development_v10 as aggregate


def fixture(tmp_path):
    frozen, trained = {}, {}
    for metric in ("i2t", "t2i", "original", "source_pair"):
        for raw in (frozen, trained):
            raw[f"{metric}_cluster_ids"] = np.array(["a", "b"])
        if metric in ("i2t", "t2i"):
            frozen[f"{metric}_correct"] = np.array([False, True])
            trained[f"{metric}_correct"] = np.array([True, True])
        else:
            frozen[f"{metric}_correct"] = np.array([0., 0.])
            trained[f"{metric}_correct"] = np.array([.5, .5]) if metric == "original" else np.array([0., 0.])
            for raw in (frozen, trained):
                raw[f"{metric}_triplet_counts"] = np.array([2, 2])
    changes, paired = official.paired_changes(frozen, trained)
    exact = official.exact_changes(paired)
    effects, samples = official.selected_uncertainty(paired)
    arrays = {"frozen": frozen, "trained": trained, "paired": paired, "bootstrap": samples}
    artifacts = {}
    for name, raw in arrays.items():
        path = tmp_path / f"{name}.npz"; np.savez_compressed(path, **raw)
        artifacts[name] = {"path": str(path), "sha256": official.digest(path)}
    gate = official.gate_checks(exact, effects, True)
    return {"artifacts": artifacts, "paired_changes": changes, "exact_paired_changes": exact, "effects": effects,
            "state": {"family": "joint", "nonzero": True}, "gate": gate, "passed": gate["passed"]}, arrays


def test_raw_bootstrap_reconstruction_uses_boolean_nonzero_not_derived_norm(tmp_path):
    result, _ = fixture(tmp_path)
    with patch.object(official, "verify_record", side_effect=lambda entry: Path(entry["path"])):
        assert official.reconstruct_gate(result)["passed"]
    assert "update_norm" not in result["state"]


def test_altered_bootstrap_draw_rejected_even_if_extreme_quantiles_unchanged(tmp_path):
    result, arrays = fixture(tmp_path)
    samples = arrays["bootstrap"]["i2t"]
    where = np.flatnonzero(samples == .5)[0]
    before = np.quantile(samples, [.05 / 160, 1 - .05 / 160])
    samples[where] += .00001
    np.testing.assert_array_equal(before, np.quantile(samples, [.05 / 160, 1 - .05 / 160]))
    path = result["artifacts"]["bootstrap"]["path"]
    np.savez_compressed(path, **arrays["bootstrap"])
    result["artifacts"]["bootstrap"]["sha256"] = official.digest(path)
    with patch.object(official, "verify_record", side_effect=lambda entry: Path(entry["path"])):
        with pytest.raises(ValueError, match="bootstrap draws"):
            official.reconstruct_gate(result)


def test_strict_boundary_equality_still_fails(tmp_path):
    result, _ = fixture(tmp_path)
    result["effects"]["i2t"]["ci_lower"] = -.01
    result["effects"]["i2t"]["exact_ci_lower"] = {"numerator": -1, "denominator": 100}
    assert not official.gate_checks(result["exact_paired_changes"], result["effects"], True)["passed"]
    result["effects"]["i2t"]["ci_lower"] = 0
    result["effects"]["i2t"]["exact_ci_lower"] = {"numerator": 0, "denominator": 1}
    result["exact_paired_changes"]["original"] = {"numerator": 0, "denominator": 1}
    assert not official.gate_checks(result["exact_paired_changes"], result["effects"], True)["passed"]


def test_fixed_seed_mean_preserves_exact_rational_cancellation(tmp_path):
    _, arrays = fixture(tmp_path)
    paired = {seed: {key: value.copy() for key, value in arrays["paired"].items()} for seed in aggregate.SEEDS}
    for seed, sign in ((17, 1), (29, -1), (43, 0)):
        paired[seed]["original_difference_numerators"][:] = sign
        paired[seed]["original_difference_denominators"][:] = [3, 7]
        paired[seed]["original_difference"] = np.array([sign / 3, sign / 7])
    mean = aggregate.fixed_seed_pairs(paired)
    exact = official.exact_changes(mean)
    assert exact["original"]["numerator"] == 0
    assert np.all(mean["original_difference"] == 0)
    effects, _ = official.selected_uncertainty(mean)
    assert aggregate.aggregate_gate(exact, effects, [True, True, True])["individual_replication_pass_required"] is False
    with pytest.raises(ValueError):
        aggregate.aggregate_gate(exact, effects, [True, False, True])


def test_development_lock_rejects_mixed_seeds_and_duplicate_encoders():
    protocol = {"study": "sanw_practical_v10", "development_contract": {
        **official.CONTRACT, "selection": "one_preselected_fullfit_state_no_development_search"},
        "development_inputs": {}, "evaluation_status": "exploratory"}
    def verify(run, *args):
        encoder, seed = run
        return None, None, {"encoder": encoder, "family": "joint", "seed": seed, "nonzero": True}
    with patch.object(official, "require_evaluation_threads"), patch.object(official, "root_path", side_effect=lambda x: x), \
         patch.object(official, "digest", return_value="hash"), patch.object(official, "read", return_value=protocol), \
         patch.object(official, "record", return_value={}), patch.object(official, "verify_full_pilot", side_effect=verify):
        good = [("vit_b32", 17), ("rn50", 17)]
        assert official.build_lock("protocol", "hash", good)["seed"] == 17
        for bad in ([good[0], ("rn50", 29)], [good[0], good[0]]):
            with pytest.raises(ValueError):
                official.build_lock("protocol", "hash", bad)
