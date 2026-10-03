"""Independent small examples for ranking, query alignment and cluster inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gcr.evaluation import (
    evaluate_relations, evaluate_retrieval, evaluate_triplets, full_pool_ranks,
    paired_image_bootstrap, ranks_to_metrics, supported_contradicted_accuracy,
)


def brute_ranks(queries, candidates, relevance):
    scores = np.asarray(queries, dtype=np.float64) @ np.asarray(candidates, dtype=np.float64).T
    ordering = [sorted(range(len(candidates)), key=lambda j: (-row[j], j)) for row in scores]
    ranks = [next(k + 1 for k, item in enumerate(order) if item in relevant)
             for order, relevant in zip(ordering, relevance)]
    return ranks, ordering


@pytest.mark.parametrize("block_size", [1, 2, 9])
def test_full_pool_exact_ranks_match_independent_stable_sort(block_size):
    # Duplicate candidates create top-k boundary ties and multi-positive ties.
    queries = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32)
    candidates = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, 1]], dtype=np.float32)
    relevance = [[1], [5], [2, 3], [0, 1]]
    expected_ranks, expected_ordering = brute_ranks(queries, candidates, relevance)
    actual = full_pool_ranks(queries, candidates, relevance, block_size=block_size, top_k=2)
    np.testing.assert_array_equal(actual["ranks"], expected_ranks)
    np.testing.assert_array_equal(actual["top_indices"], np.asarray(expected_ordering)[:, :2])
    assert actual["ranks"][0] == 2  # An equally scored earlier negative wins the tie.


def test_ranks_are_one_based_and_recall_cutoffs_exact():
    actual = ranks_to_metrics(np.array([1, 2, 5, 6, 10, 11]))
    assert actual["r1"] == 1 / 6
    assert actual["r5"] == 3 / 6
    assert actual["r10"] == 5 / 6
    assert actual["median_rank"] == 5.5
    with pytest.raises(ValueError):
        ranks_to_metrics(np.array([0]))


def test_both_retrieval_directions_use_full_many_caption_pool():
    images = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    texts = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float32)
    iid, tid = ["i0", "i1", "i2"], [f"t{i}" for i in range(6)]
    pairs = [{"image_id": iid[i // 2], "text_id": tid[i], "relation": "source"} for i in range(6)]
    metrics, predictions = evaluate_retrieval(images, texts, iid, tid, pairs, block_size=2)
    expected_i2t, _ = brute_ranks(images, texts, [[0, 1], [2, 3], [4, 5]])
    expected_t2i, _ = brute_ranks(texts, images, [[0], [0], [1], [1], [2], [2]])
    np.testing.assert_array_equal(predictions["i2t_ranks"], expected_i2t)
    np.testing.assert_array_equal(predictions["t2i_ranks"], expected_t2i)
    assert metrics["i2t"]["r1"] == 1
    assert predictions["text_source_image_ids"].tolist() == ["i0", "i0", "i1", "i1", "i2", "i2"]


def test_relation_metric_averages_images_not_pairs_and_preserves_half_ties():
    images = np.array([[1, 0], [0, 1], [-1, -1]], dtype=np.float32)
    texts = np.array([[1, 0], [0, 0], [-1, 0], [0, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32)
    iid, tid = ["a", "b", "c"], [f"t{i}" for i in range(7)]
    specs = [("a", 0, "supported"), ("a", 1, "supported"), ("a", 2, "contradicted"),
             ("a", 3, "contradicted"), ("b", 4, "contradicted"), ("b", 5, "supported"), ("c", 6, "neutral")]
    pairs = [{"image_id": image, "text_id": tid[index], "relation": relation} for image, index, relation in specs]
    metrics, predictions = evaluate_relations(images, texts, iid, tid, pairs)
    assert predictions["image_accuracy"].tolist() == [0.875, 0.0]
    assert metrics["accuracy"] == 0.4375  # Pair-weighted would be 0.7.
    assert metrics["excluded_images"] == [{"image_id": "c", "supported": 0, "contradicted": 0}]
    assert supported_contradicted_accuracy([1], [1]) == 0.5


def test_sugarcrepe_two_positive_requirement_strict_ties():
    images = np.array([[1.0], [1.0]])
    texts = np.array([[1.0], [0.0], [0.0], [2.0], [3.0], [1.0]])
    triplets = [
        {"id": "q1", "image_id": "i1", "positive1_id": "p1", "positive2_id": "p2", "negative_id": "n1", "category": "replace_obj"},
        {"id": "q2", "image_id": "i2", "positive1_id": "p3", "positive2_id": "p4", "negative_id": "n2", "category": "swap_obj"},
    ]
    metrics, predictions = evaluate_triplets(images, texts, ["i1", "i2"], ["p1", "p2", "n1", "p3", "p4", "n2"], triplets)
    assert metrics["positive1_accuracy"] == 1
    assert metrics["positive2_accuracy"] == 0.5
    assert metrics["both_accuracy"] == 0.5
    assert predictions["correct"].tolist() == [False, True]
    single = [{key: value for key, value in row.items() if key != "positive2_id"} for row in triplets]
    single_metrics, _ = evaluate_triplets(images, texts, ["i1", "i2"], ["p1", "p2", "n1", "p3", "p4", "n2"], single)
    assert single_metrics["accuracy"] == 1


def test_cluster_bootstrap_matches_independent_repeated_image_sampling():
    a = np.array([[1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]], dtype=float)
    b = np.zeros_like(a)
    ids = ["a", "b", "b", "c", "c", "c"]
    actual = paired_image_bootstrap(a, b, ids, replicates=103, seed=123, family_size=6, batch_size=13)
    rng = np.random.default_rng(123)
    cluster_indices = [[0], [1, 2], [3, 4, 5]]
    expected = []
    averaged = (a - b).mean(axis=0)
    for _ in range(103):
        draw = rng.integers(0, 3, size=3)
        items = [item for cluster in draw for item in cluster_indices[cluster]]
        expected.append(sum(float(averaged[item]) for item in items) / len(items))
    np.testing.assert_allclose(actual["bootstrap_differences"], expected, rtol=0, atol=1e-15)
    assert actual["difference"] == averaged.mean()
    assert actual["confidence"] == 1 - .05 / 6
    expected_ci = np.quantile(expected, [.05 / 12, 1 - .05 / 12], method="linear")
    np.testing.assert_allclose([actual["ci_lower"], actual["ci_upper"]], expected_ci)
    assert actual["images"] == 3 and actual["items"] == 6
    assert actual["seed_differences"] == [0.5, 1 / 3]


def test_bootstrap_rejects_misaligned_pairs():
    with pytest.raises(ValueError):
        paired_image_bootstrap(np.ones((3, 4)), np.ones((3, 3)), ["a"] * 4)


def test_independent_aggregate_audit_detects_corrupt_raw_flags():
    script_path = Path(__file__).resolve().parents[1] / "scripts/analyze_study.py"
    spec = importlib.util.spec_from_file_location("analysis_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = {"positive1_scores": np.array([1.0, 1.0]), "negative_scores": np.array([0.0, 1.0]),
                   "positive1_correct": np.array([True, False]), "correct": np.array([True, False])}
    assert module.independent_metrics("sugarcrepe", predictions)["accuracy"] == .5
    predictions["correct"][1] = True
    with pytest.raises(ValueError, match="benchmark correctness"):
        module.independent_metrics("sugarcrepe", predictions)


def test_zero_adapter_and_frozen_share_exact_final_normalization():
    from gcr.adapters import ResidualAdapter
    script_path = Path(__file__).resolve().parents[1] / "scripts/evaluate_study.py"
    spec = importlib.util.spec_from_file_location("evaluate_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(55)
    images, texts = rng.normal(size=(4, 7)).astype(np.float32), rng.normal(size=(5, 7)).astype(np.float32)
    images /= np.linalg.norm(images, axis=1, keepdims=True)
    texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    features = {"image_features": images, "text_features": texts}
    frozen = module.adapted_features(features, None, batch_size=2)
    adapted = module.adapted_features(features, ResidualAdapter(dim=7), batch_size=2)
    np.testing.assert_array_equal(frozen[0], adapted[0])
    np.testing.assert_array_equal(frozen[1], adapted[1])
