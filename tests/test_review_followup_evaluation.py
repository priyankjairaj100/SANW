"""Independent ranking and inference checks for the review follow-up only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from gcr.evaluation import evaluate_retrieval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_review_followup as evaluator
import analyze_review_followup as analysis


@pytest.mark.parametrize("block_size", [1, 2, 9])
def test_compact_retrieval_matches_stable_sort_and_original_exactly(block_size):
    images = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    texts = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 1], [-1, 0], [0, -1]], dtype=np.float32)
    iid, tid = [f"i{i}" for i in range(4)], [f"t{i}" for i in range(8)]
    # Tied relevant scores have an earlier irrelevant candidate, exposing tie policy.
    owners = [3, 0, 1, 2, 1, 2, 3, 0]
    pairs = [{"image_id": iid[owner], "text_id": tid[j], "relation": "source"} for j, owner in enumerate(owners)]
    metrics, compact = evaluator.compact_retrieval(images, texts, iid, tid, pairs, block_size)
    original_metrics, original = evaluate_retrieval(images, texts, iid, tid, pairs, block_size)
    for key, value in compact.items():
        np.testing.assert_array_equal(value, original[key])
    assert metrics["i2t"] == original_metrics["i2t"] and metrics["t2i"] == original_metrics["t2i"]
    scores = images.astype(np.float64) @ texts.astype(np.float64).T
    for direction, matrix in (("i2t", scores), ("t2i", scores.T)):
        expected = []
        for qi, row in enumerate(matrix):
            relevant = {j for j, owner in enumerate(owners) if owner == qi} if direction == "i2t" else {owners[qi]}
            ordered = sorted(range(len(row)), key=lambda j: (-row[j], j))
            expected.append(next(rank for rank, index in enumerate(ordered, 1) if index in relevant))
        np.testing.assert_array_equal(compact[f"{direction}_ranks"], expected)
    assert not any("top_" in key for key in compact)


@pytest.mark.parametrize("defect", ["missing_owner", "two_owners", "duplicate_id"])
def test_source_retrieval_rejects_incomplete_or_ambiguous_ownership(defect):
    images, texts = np.eye(2), np.eye(2)
    iid, tid = ["i0", "i1"], ["t0", "t1"]
    pairs = [{"image_id": iid[j], "text_id": tid[j], "relation": "source"} for j in range(2)]
    if defect == "missing_owner":
        pairs.pop()
    elif defect == "two_owners":
        pairs.append({"image_id": "i1", "text_id": "t0", "relation": "source"})
    else:
        tid[1] = "t0"
    with pytest.raises(ValueError):
        evaluator.compact_retrieval(images, texts, iid, tid, pairs)


def test_protocol_rejects_changed_family_or_endpoint():
    protocol = json.loads((ROOT / "docs/REVIEW_FOLLOWUP_PROTOCOL.json").read_text())
    evaluator.validate_protocol(protocol)
    for section, key, replacement in [
        ("primary_analysis", "family_size", 6), ("primary_analysis", "bootstrap_seed", 20261003),
        ("evaluation", "trajectory_epochs", [1, 10]), ("training", "seeds", [17, 29]),
        ("primary_analysis", "endpoints", [{"dataset": "coco_karpathy", "metric": "i2t.r1"}]),
    ]:
        changed = copy.deepcopy(protocol)
        changed[section][key] = replacement
        with pytest.raises(ValueError, match="protocol field"):
            evaluator.validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["primary_analysis"]["contrasts"][1]["right_conditions"] = ["count_only_draw_0"]
    with pytest.raises(ValueError, match="contrast family"):
        evaluator.validate_protocol(changed)


def synthetic_manifest():
    states = []
    for method in evaluator.POLICIES:
        draw = int(method[-1]) if "draw_" in method else None
        condition = method.split("_draw_")[0]
        for lr in evaluator.LEARNING_RATES:
            for seed in evaluator.SEEDS:
                for epoch in range(11):
                    states.append({"state_id": f"{method}_{lr}_{seed}_{epoch}", "condition": condition, "method": method,
                                   "draw_id": draw, "learning_rate": lr, "seed": seed, "epoch": epoch,
                                   "checkpoint": "unused", "checkpoint_sha256": "unused"})
    selections = []
    for selector in ("native", "source_retrieval"):
        for state in states:
            if state["learning_rate"] == .0001 and state["epoch"] == 5:
                selections.append({key: state[key] for key in ("condition", "draw_id", "learning_rate", "seed", "epoch", "state_id")} | {"selector": selector})
    return {"states": states, "selections": selections}


def test_state_plan_deduplicates_selections_and_rejects_missing_grid(monkeypatch):
    monkeypatch.setattr(evaluator, "normalized_state", lambda x: x)
    manifest = synthetic_manifest()
    terminal, _ = evaluator.plan_states(manifest, "terminal", [1, 5, 10])
    trajectory, _ = evaluator.plan_states(manifest, "trajectory", [1, 5, 10])
    selected, _ = evaluator.plan_states(manifest, "selected", [1, 5, 10])
    assert len(terminal) == 73 and len(trajectory) == 217 and len(selected) == 25
    assert len({row["state_id"] for row in selected}) == 25
    broken = copy.deepcopy(manifest)
    broken["states"].pop()
    with pytest.raises(ValueError, match="792"):
        evaluator.plan_states(broken, "terminal", [1, 5, 10])
    broken = copy.deepcopy(manifest)
    broken["selections"][0]["epoch"] = 9
    with pytest.raises(ValueError, match="Selection metadata"):
        evaluator.plan_states(broken, "selected", [1, 5, 10])


def test_primary_averages_draws_within_seed_and_clusters_text_queries(monkeypatch):
    monkeypatch.setattr(analysis, "PRIMARY_REPLICATES", 107)
    states, predictions = {}, {}
    ids, tids = np.array(["a", "b", "c"]), np.array([f"{i}_{j}" for i in "abc" for j in range(2)])
    owner = np.repeat(ids, 2)
    for row in synthetic_manifest()["states"]:
        if row["epoch"] != 10:
            continue
        sid = row["state_id"]
        states[sid] = row
        seed_index = evaluator.SEEDS.index(row["seed"])
        if row["method"] == "supported":
            correct = np.roll([1, 1, 0], seed_index)
        elif row["method"] == "source":
            correct = np.roll([0, 1, 0], seed_index)
        else:
            correct = np.roll([1, 0, 0], row["draw_id"])
        predictions[(sid, analysis.PRIMARY_DATASET)] = {"image_ids": ids, "text_ids": tids, "text_source_image_ids": owner,
                                                      "i2t_ranks": np.where(correct, 1, 2),
                                                      "t2i_ranks": np.where(np.repeat(correct, 2), 1, 2)}
    effects, arrays = analysis.primary_analysis(states, predictions)
    assert len(effects) == 18
    target = next(effect for effect in effects if effect["right"] == "count_only" and effect["learning_rate"] == .0001 and effect["metric"] == "t2i.r1")
    assert target["training_seeds"] == 3
    assert target["images"] == 3 and target["items"] == 6
    assert np.asarray(target["comparator_draw_by_seed_values"]).shape == (3, 3)
    assert target["difference"] == pytest.approx(1 / 3)
    # Every image has 2/3 support success averaged over seeds and 1/3 randomized
    # success averaged over draws. Independently resampling image clusters is constant.
    np.testing.assert_allclose(arrays[target["effect_id"]], 1 / 3, rtol=0, atol=1e-15)
    assert target["confidence"] == 1 - .05 / 18


def test_analysis_cli_resolves_relative_paths_against_repository(monkeypatch, tmp_path):
    class ReachedMerge(Exception):
        pass

    def check_paths(paths, protocol_hash):
        assert paths == [ROOT / "results/review_followup/evaluation/terminal_index.json"]
        assert str(paths[0].relative_to(ROOT)) == "results/review_followup/evaluation/terminal_index.json"
        assert protocol_hash == evaluator.sha256(ROOT / "docs/REVIEW_FOLLOWUP_PROTOCOL.json")
        raise ReachedMerge

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "merge_indices", check_paths)
    monkeypatch.setattr(sys, "argv", ["analyze_review_followup.py", "--protocol", "docs/REVIEW_FOLLOWUP_PROTOCOL.json",
                                     "--protocol-sha256", evaluator.sha256(ROOT / "docs/REVIEW_FOLLOWUP_PROTOCOL.json"),
                                     "--indices", "results/review_followup/evaluation/terminal_index.json",
                                     "--output", "results/review_followup/path_test"])
    with pytest.raises(ReachedMerge):
        analysis.main()
    assert evaluator.path_at_root(Path("results/review_followup/path_test")) == ROOT / "results/review_followup/path_test"
    assert evaluator.path_at_root(ROOT / "docs/REVIEW_FOLLOWUP_PROTOCOL.json") == ROOT / "docs/REVIEW_FOLLOWUP_PROTOCOL.json"
