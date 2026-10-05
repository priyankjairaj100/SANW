"""Synthetic checks of the separately locked v9 development wrapper."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_practical_official_development_v9 as official


def fixture(tmp_path):
    frozen, trained = {}, {}
    for name in ("i2t", "t2i", "original", "source_pair"):
        for raw in (frozen, trained):
            raw[f"{name}_cluster_ids"] = np.array(["image_a", "image_b"])
        if name in ("i2t", "t2i"):
            frozen[f"{name}_correct"] = np.array([False, True])
            trained[f"{name}_correct"] = np.array([True, True])
        else:
            frozen[f"{name}_correct"] = np.array([0., 0.])
            trained[f"{name}_correct"] = np.array([.5, .5]) if name == "original" else np.array([0., 0.])
            for raw in (frozen, trained):
                raw[f"{name}_triplet_counts"] = np.array([2, 2])
    changes, paired = official.paired_changes(frozen, trained)
    exact = official.exact_changes(paired)
    effects, bootstrap = official.selected_uncertainty(paired)
    arrays = {"frozen": frozen, "trained": trained, "paired": paired, "bootstrap": bootstrap}
    artifacts = {}
    for name, raw in arrays.items():
        filename = tmp_path/f"{name}.npz"
        np.savez_compressed(filename, **raw)
        artifacts[name] = {"path": str(filename)}
    checks = {"nonzero_trained_update": True, "original_joint_improvement": True,
              "source_pair_joint_non_decrease": True,
              "i2t_mean_non_decrease": True, "t2i_mean_non_decrease": True,
              "i2t_strict_adjusted_retention": True, "t2i_strict_adjusted_retention": True}
    return {"artifacts": artifacts, "paired_changes": changes, "exact_paired_changes": exact,
            "effects": effects, "state": {"family": "joint", "update_norm": .1},
            "gate": {"passed": True, "checks": checks}}, arrays


def test_gate_reconstructs_from_actual_paired_outcomes_and_bootstrap(tmp_path):
    result, _ = fixture(tmp_path)
    with patch.object(official, "verify_record", side_effect=lambda entry: Path(entry["path"])):
        assert official.reconstruct_gate(result)["passed"]


def test_modified_bootstrap_draw_rejected_even_when_quantiles_unchanged(tmp_path):
    result, arrays = fixture(tmp_path)
    altered = arrays["bootstrap"]["i2t"].copy()
    candidates = np.flatnonzero(altered == .5)
    altered[candidates[0]] = .500001
    tail = .05/160
    np.testing.assert_array_equal(np.quantile(altered, [tail,1-tail]),
                                  np.quantile(arrays["bootstrap"]["i2t"], [tail,1-tail]))
    arrays["bootstrap"]["i2t"] = altered
    np.savez_compressed(result["artifacts"]["bootstrap"]["path"], **arrays["bootstrap"])
    with patch.object(official, "verify_record", side_effect=lambda entry: Path(entry["path"])):
        with pytest.raises(ValueError):
            official.reconstruct_gate(result)


def test_official_development_lock_requires_one_family_seed_and_both_encoders():
    protocol = {"study": "sanw_practical_v9", "development_contract": {
        **official.CONTRACT, "selection": "one_preselected_fullfit_state_no_development_search"},
        "source_sha256": {}, "development_inputs": {}, "evaluation_status": "exploratory"}

    def verify(run, *args):
        encoder, family, seed = run
        return None, None, None, {"encoder": encoder, "family": family, "seed": seed}

    with patch.object(official, "root_path", side_effect=lambda x: x), \
         patch.object(official, "digest", return_value="hash"), \
         patch.object(official, "read", return_value=protocol), \
         patch.object(official, "record", return_value={}), \
         patch.object(official, "verify_full_pilot", side_effect=verify):
        good = [("vit_b32", "joint", 17), ("rn50", "joint", 17)]
        assert official.build_lock("protocol", "hash", "selection", good)["seed"] == 17
        for bad in ([good[0], ("rn50", "joint", 29)], [good[0], ("rn50", "labclip", 17)], [good[0], good[0]]):
            with pytest.raises(ValueError):
                official.build_lock("protocol", "hash", "selection", bad)
