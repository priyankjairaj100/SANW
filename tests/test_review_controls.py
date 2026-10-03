"""Invariant and gradient checks for immutable promotion controls."""
import copy
import itertools
import json

import numpy as np
import pytest
import torch

from gcr.losses import contrastive_loss, policy_components, symmetric_weighted_loss
from gcr.review_controls import (
    ASSIGNMENT_SEEDS, CONTROLS, apply_promotion_assignment,
    assignment_content_sha256, batch_positive_mask, load_assignment,
    make_promotion_assignment, save_assignment, validate_assignment,
)


PROVENANCE = {"manifest_sha256": "1" * 64, "features_sha256": "2" * 64, "protocol_sha256": "3" * 64}


def fixture():
    image_features = np.asarray([[1., 0.], [0., 1.], [2 ** -.5, 2 ** -.5]])
    image_ids = ["image:0", "image:1", "image:2"]
    text_ids, text_features, relations = [], [], []
    hypothesis_scores = [[-.9, -.6, -.1, .2, .5, .7, .9], [-.8, -.4, 0., .3, .6, .95], []]
    hypothesis_labels = [[2, 3, 2, 4, 3, 4, 2], [3, 2, 4, 2, 3, 4], []]
    for image in range(3):
        own = {}
        for index in range(5):
            own[len(text_ids)] = 1
            text_ids.append(f"source:{image}:{index}")
            text_features.append(image_features[image])
        for index, (score, label) in enumerate(zip(hypothesis_scores[image], hypothesis_labels[image])):
            own[len(text_ids)] = label
            text_ids.append(f"hypothesis:{image}:{index}")
            vector = [score, np.sqrt(1 - score * score)] if image == 0 else [np.sqrt(1 - score * score), score]
            text_features.append(vector)
        relations.append(own)
    return dict(image_ids=image_ids, text_ids=text_ids, image_features=image_features,
                text_features=np.asarray(text_features), text_strings=text_ids.copy(), relations_by_image=relations,
                training_image_indices=[0, 1, 2], provenance=PROVENANCE.copy())


def assignment(control="count_only", seed=101, inputs=None):
    return make_promotion_assignment(**(fixture() if inputs is None else inputs), control=control, assignment_seed=seed)


def batch(inputs, images):
    columns = sorted({j for image in images for j in inputs["relations_by_image"][image]})
    positions = {j: k for k, j in enumerate(columns)}
    relations = torch.zeros((len(images), len(columns)), dtype=torch.int64)
    for row, image in enumerate(images):
        for j, code in inputs["relations_by_image"][image].items():
            relations[row, positions[j]] = code
    return relations, list(images), columns


@pytest.mark.parametrize("control", CONTROLS)
def test_assignments_reproduce_independently_of_training_index_order(control):
    inputs = fixture()
    a = assignment(control, inputs=inputs)
    inputs["training_image_indices"] = [2, 0, 1]
    b = assignment(control, inputs=inputs)
    assert a == b
    assert a["assignment_content_sha256"] == assignment_content_sha256(a)
    assert a["training_image_indices"] == [0, 1, 2]
    assert a["summary"]["source_positives"] == 15
    assert a["summary"]["promoted_hypotheses"] == 5


@pytest.mark.parametrize("control", CONTROLS)
def test_all_three_declared_draws_are_distinct_and_fixed(control):
    draws = [assignment(control, seed) for seed in ASSIGNMENT_SEEDS]
    patterns = [tuple(tuple(row["promoted_text_indices"]) for row in draw["records"]) for draw in draws]
    assert len(set(patterns)) == 3
    for seed, draw in zip(ASSIGNMENT_SEEDS, draws):
        assert assignment(control, seed) == draw


def test_count_only_samples_the_whole_own_hypothesis_pool():
    inputs = fixture()
    record = assignment("count_only", 101, inputs)
    rng = np.random.Generator(np.random.PCG64(101))
    for image in record["records"]:
        candidates = sorted(j for j, code in inputs["relations_by_image"][image["image_index"]].items() if code != 1)
        expected = sorted(map(int, rng.choice(np.asarray(candidates, dtype=np.int64), size=image["promotion_count"], replace=False)))
        assert image["promoted_text_indices"] == expected
    # Assignments may deliberately include contradicted or neutral reference labels.
    assert any(row["changed_promotions"] > 0 for row in record["records"])


def test_score_stratification_preserves_each_bin_count_and_odd_bin_size():
    record = assignment("score_stratified")
    for row in record["records"]:
        bins = row["score_bins"]
        assert len(bins[0]["hypothesis_text_indices"]) == (row["hypothesis_count"] + 1) // 2
        assert len(bins[1]["hypothesis_text_indices"]) == row["hypothesis_count"] // 2
        for bucket in bins:
            assert bucket["supported_count"] == bucket["promoted_count"]
    assert [bucket["supported_count"] for bucket in record["records"][0]["score_bins"]] == [2, 1]


def test_equal_cosine_ties_use_manifest_text_order():
    inputs = fixture()
    hypothesis_indices = [j for j, code in inputs["relations_by_image"][0].items() if code != 1]
    inputs["text_features"][hypothesis_indices] = [0., 1.]
    record = assignment("score_stratified", inputs=inputs)["records"][0]
    assert record["score_bins"][0]["hypothesis_text_indices"] == hypothesis_indices[:4]
    assert record["score_bins"][1]["hypothesis_text_indices"] == hypothesis_indices[4:]


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("all_supported", [False, True])
def test_zero_and_all_supported_boundaries_have_no_spurious_changes(control, all_supported):
    inputs = fixture()
    own = inputs["relations_by_image"][0]
    for j in own:
        if own[j] != 1:
            own[j] = 2 if all_supported else 3
    row = assignment(control, inputs=inputs)["records"][0]
    assert row["promotion_count"] == (7 if all_supported else 0)
    assert row["changed_promotions"] == 0
    assert row["source_target_share"] == 5 / (12 if all_supported else 5)
    assert row["promoted_text_indices"] == row["supported_text_indices"]


def test_diagnostics_measure_real_assignment_change_and_score_mismatch():
    result = assignment("score_stratified")
    changed_total = 0
    for row in result["records"]:
        chosen, true = set(row["promoted_text_indices"]), set(row["supported_text_indices"])
        changed = len(chosen - true)
        assert row["label_overlap_count"] == len(chosen & true)
        assert row["changed_promotions"] == changed
        assert row["symmetric_difference_count"] == len(chosen ^ true)
        assert sum(row["promoted_reference_label_counts"].values()) == len(chosen)
        scores = {hyp["text_index"]: hyp["frozen_cosine"] for hyp in row["hypotheses"]}
        if chosen:
            expected = np.sort([scores[j] for j in chosen]) - np.sort([scores[j] for j in true])
            assert row["frozen_score_diagnostics"]["mean_absolute_sorted_difference"] == pytest.approx(np.abs(expected).mean())
        changed_total += changed
    assert result["summary"]["changed_promotions"] == changed_total
    # Histogram matching does not claim exact continuous-score matching.
    assert result["summary"]["frozen_score_diagnostics"]["max_absolute_sorted_difference"] > 0


@pytest.mark.parametrize("control", CONTROLS)
def test_every_image_subset_preserves_pool_positive_counts_and_reverse_anchor_counts(control):
    inputs = fixture()
    record = assignment(control, inputs=inputs)
    for size in range(1, 4):
        for images in itertools.combinations(range(3), size):
            relations, images, texts = batch(inputs, images)
            original = relations.clone()
            transformed = apply_promotion_assignment(relations, images, texts, record)
            assert torch.equal(relations, original)
            assert transformed.shape == relations.shape
            assert torch.equal(transformed == 1, relations == 1)
            true_positive = (relations == 1) | (relations == 2)
            new_positive = (transformed == 1) | (transformed == 2)
            assert torch.equal(true_positive.sum(1), new_positive.sum(1))
            assert true_positive.any(0).sum() == new_positive.any(0).sum()
            assert torch.all(new_positive.sum(0)[new_positive.any(0)] == 1)
            assert set(transformed.flatten().tolist()) <= {0, 1, 2}


def test_source_score_gradients_match_in_both_directions_at_identical_logits():
    inputs = fixture()
    record = assignment("score_stratified", inputs=inputs)
    relations, images, texts = batch(inputs, [0, 1])
    transformed = apply_promotion_assignment(relations, images, texts, record)
    logits = torch.linspace(-1.1, 1.7, relations.numel(), dtype=torch.float64).reshape(relations.shape).requires_grad_()
    original = symmetric_weighted_loss(logits, policy_components(relations, "multipositive", dtype=logits.dtype))
    randomized = symmetric_weighted_loss(logits, policy_components(transformed, "multipositive", dtype=logits.dtype))
    a, = torch.autograd.grad(original, logits)
    b, = torch.autograd.grad(randomized, logits)
    source_columns = (relations == 1).any(0)
    torch.testing.assert_close(a[:, source_columns], b[:, source_columns], atol=1e-14, rtol=1e-14)
    assert not torch.allclose(a, b)


def test_existing_multipositive_loss_handles_transformed_batch_and_gradients():
    inputs = fixture()
    record = assignment("count_only", inputs=inputs)
    relations, images, texts = batch(inputs, [1, 0])
    transformed = apply_promotion_assignment(relations, images, texts, record)
    image_features = torch.tensor(inputs["image_features"][images], dtype=torch.float64, requires_grad=True)
    text_features = torch.tensor(inputs["text_features"][texts], dtype=torch.float64, requires_grad=True)
    result = contrastive_loss(image_features, text_features, transformed, "multipositive", 3.)
    expected = symmetric_weighted_loss(3 * image_features @ text_features.T,
                                      policy_components(transformed, "multipositive", dtype=torch.float64))
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.isfinite(image_features.grad).all() and torch.isfinite(text_features.grad).all()
    assert image_features.grad.norm() > 0 and text_features.grad.norm() > 0


def test_batch_index_permutations_preserve_the_same_assignment():
    inputs = fixture()
    record = assignment("count_only", inputs=inputs)
    relations, images, texts = batch(inputs, [0, 1])
    a = batch_positive_mask(relations, images, texts, record)
    order = list(reversed(range(len(texts))))
    b = batch_positive_mask(relations.flip(0)[:, order], list(reversed(images)), [texts[j] for j in order], record)
    assert torch.equal(a.flip(0)[:, order], b)


def test_shared_candidate_ownership_is_rejected_instead_of_claiming_anchor_matching():
    inputs = fixture()
    shared = next(j for j, code in inputs["relations_by_image"][0].items() if code != 1)
    inputs["relations_by_image"][1][shared] = 3
    with pytest.raises(ValueError, match="one-image-owned"):
        assignment(inputs=inputs)


def test_missing_batch_candidates_or_wrong_labels_are_rejected():
    inputs = fixture()
    record = assignment(inputs=inputs)
    relations, images, texts = batch(inputs, [0])
    with pytest.raises(ValueError, match="omitted"):
        batch_positive_mask(relations[:, :-1], images, texts[:-1], record)
    wrong = relations.clone()
    wrong[wrong == 2] = 3
    with pytest.raises(ValueError, match="labels disagree"):
        batch_positive_mask(wrong, images, texts, record)


def test_save_load_are_immutable_and_bound_to_file_content_and_inputs(tmp_path):
    record = assignment()
    path = tmp_path / "assignment.json"
    file_hash = save_assignment(path, record)
    assert save_assignment(path, record) == file_hash
    loaded = load_assignment(path, expected_file_sha256=file_hash, expected_provenance=PROVENANCE,
                             expected_control="count_only", expected_seed=101)
    assert loaded == record
    with pytest.raises(ValueError, match="different frozen assignment"):
        save_assignment(path, assignment(seed=211))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_assignment(path, expected_provenance={"features_sha256": "f" * 64})
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_assignment(path, expected_file_sha256=file_hash)


def test_tampered_assignment_hash_and_recomputed_hash_with_invalid_counts_are_rejected():
    record = assignment()
    tampered = copy.deepcopy(record)
    tampered["records"][0]["promoted_text_indices"] = []
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_assignment(tampered)
    tampered["assignment_content_sha256"] = assignment_content_sha256(tampered)
    with pytest.raises(ValueError, match="promotion counts"):
        validate_assignment(tampered)


def test_assignment_creation_excludes_nontraining_images_and_requires_input_hashes():
    inputs = fixture()
    inputs["training_image_indices"] = [1]
    record = assignment(inputs=inputs)
    assert [row["image_id"] for row in record["records"]] == ["image:1"]
    assert record["summary"]["promoted_hypotheses"] == 2
    inputs["provenance"] = {"manifest_sha256": "1" * 64}
    with pytest.raises(ValueError, match="features_sha256"):
        assignment(inputs=inputs)


def test_exact_string_diagnostics_distinguish_duplicate_id_swaps_from_text_changes():
    inputs = fixture()
    own = inputs["relations_by_image"][0]
    hypothesis_indices = sorted(j for j, code in own.items() if code != 1)
    first, second = hypothesis_indices[:2]
    inputs["relations_by_image"][0] = {j: code for j, code in own.items() if code == 1 or j in (first, second)}
    inputs["relations_by_image"][0][first] = 2
    inputs["relations_by_image"][0][second] = 3
    inputs["text_strings"][first] = inputs["text_strings"][second] = "The same literal hypothesis."
    inputs["text_features"][[first, second]] = [0., 1.]
    # Select a constructed draw that swaps IDs; the production sampler never retries.
    rows = [assignment("count_only", seed, inputs)["records"][0] for seed in range(10)]
    row = next(value for value in rows if value["changed_promotions"] == 1)
    assert row["exact_string_changed_promotions"] == 0
    assert row["exact_string_overlap_count"] == 1
    assert not row["exact_string_changed_image"]
    assert row["duplicate_hypothesis_string_groups"] == 1
    assert row["duplicate_hypothesis_strings_beyond_first"] == 1
    assert len(row["exact_score_tie_groups"]) == 1
    assert row["exact_score_ties_beyond_first"] == 1
    assert row["score_tie_crosses_bin_boundary"]


def test_forced_score_bins_are_reported_without_resampling_until_changed():
    inputs = fixture()
    own = inputs["relations_by_image"][0]
    hypotheses = sorted(j for j, code in own.items() if code != 1)
    for rank, j in enumerate(hypotheses):
        own[j] = 2 if rank < 4 else 3
    row = assignment("score_stratified", inputs=inputs)["records"][0]
    assert not row["assignment_exchangeable"]
    assert row["possible_assignments"] == 1
    assert row["changed_promotions"] == 0
    assert all(not unit["exchangeable"] for unit in row["sampling_units"])
    unrestricted = assignment("count_only", inputs=inputs)["records"][0]
    assert unrestricted["assignment_exchangeable"]
    assert unrestricted["possible_assignments"] == 35


def test_exact_text_diagnostics_require_manifest_strings():
    inputs = fixture()
    inputs["text_strings"] = inputs["text_strings"][:-1]
    with pytest.raises(ValueError, match="exact manifest string"):
        assignment(inputs=inputs)
