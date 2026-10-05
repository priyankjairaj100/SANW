from types import SimpleNamespace

import numpy as np
import pytest

from gcr.practical_constrained_evaluation_v8 import (
    CanonicalScorer, canonical_pair_scores, composition_metrics,
    development_gate, exact_retrieval, paired_cluster_bootstrap,
)
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer


def fixture_scorer(dimension=19, rank=5):
    rng = np.random.default_rng(931)
    U = np.linalg.qr(rng.normal(size=(dimension, rank)))[0]
    V = np.linalg.qr(rng.normal(size=(dimension, rank)))[0]
    state = [rng.normal(size=dimension) * .03, rng.normal(size=dimension) * .02,
             U, V, rng.normal(size=(rank, rank)) * .01]
    return CanonicalScorer(*state), ConstrainedBilinearScorer(*state)


def exhaustive(images, texts, scorer):
    scores = np.empty((len(images), len(texts)))
    for i, image in enumerate(images):
        scores[i] = scorer.pair_scores(np.repeat(image[None], len(texts), axis=0), texts)
    return scores


def test_canonical_pair_batch_permutation_and_training_scorer_agree():
    scorer, training_scorer = fixture_scorer()
    rng = np.random.default_rng(552)
    images, texts = rng.normal(size=(31, 19)), rng.normal(size=(31, 19))
    images[17], texts[17] = images[2], texts[2]
    all_scores = scorer.pair_scores(images, texts)
    one_at_a_time = np.concatenate([scorer.pair_scores(images[i:i+1], texts[i:i+1]) for i in range(len(images))])
    order = rng.permutation(len(images))
    assert np.array_equal(all_scores, one_at_a_time)
    assert np.array_equal(scorer.pair_scores(images[order], texts[order]), all_scores[order])
    assert np.array_equal(all_scores, training_scorer.score_pairs(images, texts))
    assert all_scores[17] == all_scores[2]


@pytest.mark.parametrize("query_block", [1, 4, 64])
def test_retrieval_matches_full_canonical_matrix_with_duplicate_and_near_ties(query_block):
    scorer, _ = fixture_scorer()
    rng = np.random.default_rng(529)
    images, texts = rng.normal(size=(7, 19)), rng.normal(size=(15, 19))
    images[5] = images[2]
    texts[10] = texts[3]
    texts[11] = np.nextafter(texts[3], np.inf)
    owner = np.arange(15) % 7
    expected = exhaustive(images, texts, scorer)
    _, raw = exact_retrieval(images, texts, owner, scorer, query_block=query_block)
    assert np.array_equal(raw["i2t_top_indices"], expected.argmax(axis=1))
    assert np.array_equal(raw["t2i_top_indices"], expected.argmax(axis=0))
    assert np.array_equal(raw["i2t_top_scores"], expected[np.arange(7), expected.argmax(axis=1)])
    assert np.array_equal(raw["t2i_top_scores"], expected[expected.argmax(axis=0), np.arange(15)])


def test_frozen_exact_ties_take_first_manifest_index():
    images = np.array([[1., 0.], [1., 0.]])
    texts = np.array([[1., 0.], [1., 0.], [0., 1.]])
    _, raw = exact_retrieval(images, texts, np.array([0, 1, 1]))
    assert raw["i2t_top_indices"].tolist() == [0, 0]
    assert raw["t2i_top_indices"].tolist() == [0, 0, 0]
    assert raw["i2t_candidate_counts"].tolist() == [2, 2]
    assert raw["t2i_candidate_counts"].tolist() == [2, 2, 2]


def test_composition_all_triples_image_weight_and_source_conflicts():
    # Image 0 has an exact source/negative conflict under distinct IDs.
    # It remains in the original metric, and is excluded only from source-pair.
    data = SimpleNamespace(
        images=np.array([[1., 0.], [0., 1.]]),
        texts=np.array([[.9, 0], [.8, 0], [.7, 0], [.6, 0], [.9, 0],
                        [0, .5], [0, .4], [0, .2], [0, .3]]),
        image_ids=["i0", "i1"], split_indices={"validation": [0, 1]},
        pairs=[{0: 1, 1: 1, 2: 2, 3: 3, 4: 3}, {5: 1, 6: 1, 7: 2, 8: 3}],
        manifest={"texts": [{"text": s} for s in ["Year 1800", "Year 1900", "supported", "negative", "YEAR 1800!",
                                                    "source a", "source b", "support", "contradiction"]]})
    summary, raw = composition_metrics(data)
    assert summary["original"]["joint_accuracy"] == .25  # (.5 + 0) / 2
    assert summary["source_pair"]["joint_accuracy"] == 1
    assert summary["source_pair"]["excluded_conflicting_negatives"] == 1
    assert summary["original"]["triplet_count"] == 6
    assert raw["source_pair_triplet_counts"].tolist() == [1, 1]


def test_cluster_bootstrap_preserves_clusters_and_constant_effect():
    summary, samples = paired_cluster_bootstrap(np.ones(10) * .2, np.repeat(np.arange(5), 2), replicates=1000)
    assert np.allclose(samples, .2, rtol=0, atol=1e-15)
    assert summary["image_clusters"] == 5
    assert summary["ci_lower"] == pytest.approx(.2)
    # An uneven cluster has a resampled ratio, not the mean of image means.
    summary, samples = paired_cluster_bootstrap([1, 0, 0, 0], ["a", "b", "b", "b"], replicates=1000)
    assert set(samples) <= {0., .25, 1.}
    assert summary["difference"] == .25
    assert summary["algorithm"] == "direct_paired_image_cluster_ratio"


def test_gate_strict_boundary_and_nonzero_state_requirements():
    frozen = {"original": {"joint_accuracy": .5}, "source_pair": {"joint_accuracy": .8}}
    trained = {"original": {"joint_accuracy": .51}, "source_pair": {"joint_accuracy": .8}}
    effect = {"replicates": 100000, "bootstrap_seed": 20261007, "family_size": 80,
              "difference": 0., "ci_lower": -.01, "ci_upper": .01}
    retention = {"i2t": effect.copy(), "t2i": effect.copy()}
    assert not development_gate(frozen, trained, retention, update_norm=.01)["passed"]
    for value in retention.values():
        value["ci_lower"] = np.nextafter(-.01, np.inf)
    assert development_gate(frozen, trained, retention, update_norm=.01)["passed"]
    assert not development_gate(frozen, trained, retention, update_norm=0.)["passed"]
    trained["source_pair"]["joint_accuracy"] = .799
    assert not development_gate(frozen, trained, retention, update_norm=.01)["passed"]


def test_invalid_ownership_and_nonfinite_state_are_rejected():
    with pytest.raises(ValueError, match="ownership"):
        exact_retrieval(np.eye(2), np.eye(2), [-1, 0])
    scorer, _ = fixture_scorer()
    with pytest.raises(ValueError, match="finite"):
        scorer.pair_scores(np.full((1, 19), np.nan), np.zeros((1, 19)))
