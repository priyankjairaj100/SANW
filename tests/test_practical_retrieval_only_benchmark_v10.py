"""Toy-only release ordering, fixed supplementary coverage, and artifact checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_practical_retrieval_only_benchmark_v10 as supplement


def fake_record(path):
    return {"path": str(path), "sha256": "hash", "bytes": 1}


def test_supplement_lock_requires_exact_six_states_without_core_cycle():
    core_lock = {"protocol": {"sha256": "protocol"}, "inputs": {}, "evaluation_status": "exploratory"}
    contract = {"inherited_protocol": {"sha256": "protocol"}, "contrasts": [], "endpoints": {}, "inference": {}}
    audit = {"study": "sanw_practical_v10_independent_retrieval_only_benchmark_source_audit", "passed": True,
             "blocking_findings": [], "contract": {"sha256": "hash"}, "sources": {name: "hash" for name in supplement.SOURCES}}
    def state(run, *args):
        encoder, seed = run
        return {"encoder": encoder, "seed": seed, "family": "retrieval_only", "nonzero": True}
    runs = [(encoder, seed) for encoder in supplement.ENCODERS for seed in supplement.SEEDS]
    with patch.object(supplement.core, "load_lock", return_value=core_lock), \
         patch.object(supplement, "verify_contract", return_value=(contract, {}, None)), \
         patch.object(supplement, "read", return_value=audit), patch.object(supplement, "digest", return_value="hash"), \
         patch.object(supplement, "root_path", side_effect=Path), patch.object(supplement, "record", side_effect=fake_record), \
         patch.object(supplement, "verify_state", side_effect=state):
        lock = supplement.build_lock("core", "hash", "contract", "audit", runs)
        assert len(lock["states"]) == 6 and lock["core_lock"]["path"] == "core"
        assert lock["effect_count"] == 20 and lock["candidate_selection_allowed"] is False
        for invalid in (runs[:-1], runs[:-1] + [runs[0]]):
            with pytest.raises(ValueError):
                supplement.build_lock("core", "hash", "contract", "audit", invalid)


def test_release_requires_empty_distinct_roots_and_binds_all_optional_states(tmp_path):
    args = SimpleNamespace(core_output_root=str(tmp_path / "core"), supplement_output_root=str(tmp_path / "extra"),
                           lock="lock", output="release", additional_locks=[])
    lock = {"core_lock": fake_record("core_lock")}
    written = []
    with patch.object(supplement, "root_path", side_effect=Path), patch.object(supplement, "ROOT", tmp_path), \
         patch.object(supplement, "record", side_effect=fake_record), \
         patch.object(supplement, "write_json", side_effect=lambda p, value: written.append(value) or fake_record(p)), \
         patch.object(supplement, "verify_additional_lock", return_value={"family": "labclip", "output_root": str(tmp_path / "labclip")}):
        supplement.release(args, lock)
        assert written[-1]["selected_states_frozen_before_scoring"] == 18
        args.additional_locks = ["labclip_lock"]
        supplement.release(args, lock)
        assert written[-1]["selected_states_frozen_before_scoring"] == 24
        assert written[-1]["additional_locks"] == [fake_record("labclip_lock")]
        args.additional_locks = ["labclip_lock", "labclip_lock"]
        with pytest.raises(ValueError, match="Duplicate"):
            supplement.release(args, lock)
        args.additional_locks = []
        (tmp_path / "core").mkdir(); (tmp_path / "core" / "already_scored").write_text("outcome")
        with pytest.raises(ValueError, match="precede"):
            supplement.release(args, lock)


def test_score_core_delegates_to_frozen_evaluator_and_creates_release_chain(tmp_path):
    args = SimpleNamespace(mode="score-core", encoder="vit_b32", dataset="e_vil_test1000", release="release")
    release = {"core_output_root": str(tmp_path / "core")}
    lock = {"core_lock": fake_record("corelock")}
    written = []
    with patch.object(supplement, "root_path", side_effect=Path), patch.object(supplement, "record", side_effect=fake_record), \
         patch.object(supplement, "verify_record", side_effect=lambda e: e["path"]), patch.object(supplement, "read", return_value={}), \
         patch.object(supplement, "write_json", side_effect=lambda p, value: written.append(value) or fake_record(p)), \
         patch.object(supplement.core, "evaluate") as delegated, patch.object(supplement.core, "load_dataset") as loader:
        supplement.score(args, release, lock)
        delegated.assert_called_once()
        assert delegated.call_args.args[0].lock == "corelock"
        assert written[0]["release"] == fake_record("release") and written[0]["operation"] == "core"
        loader.assert_not_called()


def toy_raw(dataset, correct):
    if dataset == "sugarcrepe_pp":
        return {"item_ids": np.array(["one", "two"]), "image_ids": np.array(["a", "a"]),
                "categories": np.array(["toy", "toy"]), "positive1_ids": np.array(["p1", "p1"]),
                "positive2_ids": np.array(["p2", "p2"]), "negative_ids": np.array(["n", "n"]),
                "correct": np.array(correct, dtype=bool)}
    return {"image_ids": np.array(["a", "b"]), "text_ids": np.array(["one", "two"]),
            "text_source_image_ids": np.array(["a", "b"]), "owner": np.array([0, 1]),
            "i2t_correct": np.array(correct, dtype=bool), "t2i_correct": np.array(correct, dtype=bool)}


def test_all_twenty_effects_reported_even_when_core_candidate_is_worse(tmp_path):
    lock = {"core_lock": fake_record("core"), "contrasts": [["joint_minus_retrieval_only", "joint", "retrieval_only"],
             ["retrieval_only_minus_frozen", "retrieval_only", "frozen"]],
            "endpoints": {"e_vil_test1000": ["i2t.r1", "t2i.r1"], "coco_karpathy": ["i2t.r1", "t2i.r1"],
                          "sugarcrepe_pp": ["both_accuracy"]}, "evaluation_status": "exploratory"}
    release = {"supplement_lock": fake_record("supplement")}
    def prediction(release_record, release_path, lock, operation, encoder, dataset):
        if operation == "core":
            raw = {"frozen": toy_raw(dataset, [False, False])}
            raw.update({f"joint_{seed}": toy_raw(dataset, [False, False]) for seed in supplement.SEEDS})
        else:
            raw = {f"retrieval_only_{seed}": toy_raw(dataset, [True, True]) for seed in supplement.SEEDS}
        return raw, fake_record("index"), fake_record("receipt")
    written = []
    with patch.object(supplement, "load_prediction_index", side_effect=prediction), \
         patch.object(supplement, "root_path", side_effect=Path), patch.object(supplement, "record", side_effect=fake_record), \
         patch.object(supplement, "write_npz", side_effect=lambda p, value: fake_record(p)), \
         patch.object(supplement, "write_json", side_effect=lambda p, value: written.append(value) or fake_record(p)):
        supplement.analyze(SimpleNamespace(output=str(tmp_path / "analysis"), release="release"), release, lock)
    result = written[-1]
    assert result["effect_count"] == 20 and len(result["effects"]) == 20
    assert result["new_gate"] is None and result["candidate_selection_allowed"] is False
    assert result["core_practical_result_used_to_filter_effects"] is False
    assert sum(row["difference"] == -1 for row in result["effects"]) == 10
    assert sum(row["difference"] == 1 for row in result["effects"]) == 10


def test_index_cannot_be_adopted_without_matching_release_hash(tmp_path):
    release = {"core_output_root": str(tmp_path)}
    bad = {"study": "sanw_practical_v10_released_benchmark_run", "operation": "core", "release": fake_record("old_release"),
           "encoder": "vit_b32", "dataset": "e_vil_test1000", "index": fake_record("index")}
    with patch.object(supplement, "root_path", side_effect=Path), patch.object(supplement, "read", return_value=bad), \
         patch.object(supplement, "record", side_effect=fake_record):
        with pytest.raises(ValueError, match="prescoring release"):
            supplement.load_prediction_index(release, "new_release", {}, "core", "vit_b32", "e_vil_test1000")
