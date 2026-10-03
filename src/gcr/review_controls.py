"""Immutable, count-matched promotion controls for the follow-up protocol.

This module changes only which existing hypothesis candidates are positive.
It never trains a model, changes frozen losses, removes candidates, or changes
source-caption targets. Assignments are generated once, independently of
optimizer seeds, learning rates and epochs, then bound to input/source hashes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

CONTROLS = ("count_only", "score_stratified")
ASSIGNMENT_SEEDS = (101, 211, 307)
SOURCE, SUPPORTED, CONTRADICTED, NEUTRAL = 1, 2, 3, 4
RELATION_NAMES = {SUPPORTED: "supported", CONTRADICTED: "contradicted", NEUTRAL: "neutral"}
REQUIRED_PROVENANCE = ("manifest_sha256", "features_sha256", "protocol_sha256")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment_content_sha256(assignment: Mapping[str, Any]) -> str:
    """Hash every assignment field except the self-referential digest field."""
    payload = {key: value for key, value in assignment.items() if key != "assignment_content_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _check_provenance(provenance: Mapping[str, str]) -> None:
    for name in REQUIRED_PROVENANCE:
        value = provenance.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"provenance requires a lowercase SHA256 for {name}")


def _numpy_features(value: Tensor | np.ndarray, count: int, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != count or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name} feature matrix")
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1., atol=2e-4, rtol=2e-4):
        raise ValueError(f"{name} features must be nonzero and L2 normalized")
    return result / norms[:, None]


def _score_diagnostics(true_scores: Sequence[float], chosen_scores: Sequence[float]) -> dict[str, Any]:
    a, b = np.asarray(true_scores, dtype=float), np.asarray(chosen_scores, dtype=float)
    if len(a) != len(b):
        raise ValueError("Matched promotions must contain equally many scores")
    if not len(a):
        return {"count": 0, "supported_mean": None, "promoted_mean": None, "mean_difference": None,
                "mean_absolute_sorted_difference": None, "max_absolute_sorted_difference": None}
    difference = np.sort(b) - np.sort(a)
    return {"count": len(a), "supported_mean": float(a.mean()), "promoted_mean": float(b.mean()),
            "mean_difference": float(b.mean() - a.mean()),
            "mean_absolute_sorted_difference": float(np.abs(difference).mean()),
            "max_absolute_sorted_difference": float(np.abs(difference).max())}


def make_promotion_assignment(
    image_ids: Sequence[str],
    text_ids: Sequence[str],
    image_features: Tensor | np.ndarray,
    text_features: Tensor | np.ndarray,
    relations_by_image: Sequence[Mapping[int, int]],
    training_image_indices: Sequence[int],
    control: str,
    assignment_seed: int,
    provenance: Mapping[str, str],
    text_strings: Sequence[str],
) -> dict[str, Any]:
    """Generate one complete training assignment, with global manifest indices.

    ``count_only`` uniformly samples M_i of image i's own hypothesis candidates.
    ``score_stratified`` sorts those candidates by frozen image-text cosine,
    breaking ties by global text index, partitions them into two bins (lower
    bin receives the extra candidate if odd), and uniformly samples the true
    supported count in each bin. Sources always remain positive.

    One-image ownership of all annotated candidate IDs is required. Combined
    with per-image promotion-count matching, this proves exact preservation of
    newly eligible reverse anchors for any batch of these training images.
    """
    if control not in CONTROLS:
        raise ValueError(f"Unknown promotion control: {control}")
    if not isinstance(assignment_seed, int) or isinstance(assignment_seed, bool) or assignment_seed < 0:
        raise ValueError("assignment_seed must be a nonnegative integer")
    _check_provenance(provenance)
    image_ids, text_ids = [str(value) for value in image_ids], [str(value) for value in text_ids]
    text_strings = list(text_strings)
    if len(text_strings) != len(text_ids) or any(not isinstance(value, str) for value in text_strings):
        raise ValueError("text_strings must contain the exact manifest string for every text ID")
    if len(set(image_ids)) != len(image_ids) or len(set(text_ids)) != len(text_ids):
        raise ValueError("Image and text IDs must be unique")
    if len(relations_by_image) != len(image_ids):
        raise ValueError("relations_by_image must have one mapping per image")
    given_indices = [int(index) for index in training_image_indices]
    if not given_indices or len(set(given_indices)) != len(given_indices) or any(i < 0 or i >= len(image_ids) for i in given_indices):
        raise ValueError("Training image indices must be nonempty, unique and in range")
    indices = sorted(given_indices)
    image_vectors = _numpy_features(image_features, len(image_ids), "image")
    text_vectors = _numpy_features(text_features, len(text_ids), "text")
    if image_vectors.shape[1] != text_vectors.shape[1]:
        raise ValueError("Image and text feature dimensions must agree")
    owners: dict[int, int] = {}
    own_relations = {}
    for image_index in indices:
        pairs = {int(j): int(code) for j, code in relations_by_image[image_index].items()}
        if any(j < 0 or j >= len(text_ids) for j in pairs) or any(code not in (SOURCE, SUPPORTED, CONTRADICTED, NEUTRAL) for code in pairs.values()):
            raise ValueError("Own-pool annotations must contain valid text indices and relation codes 1..4")
        if SOURCE not in pairs.values():
            raise ValueError("Each training image must have a source-positive caption")
        for j in pairs:
            if j in owners:
                raise ValueError("Promotion controls require one-image-owned text IDs to preserve reverse-anchor counts")
            owners[j] = image_index
        own_relations[image_index] = pairs

    rng = np.random.Generator(np.random.PCG64(assignment_seed))
    records = []
    pooled_true_scores, pooled_chosen_scores = [], []
    for image_index in indices:
        pairs = own_relations[image_index]
        sources = sorted(j for j, code in pairs.items() if code == SOURCE)
        hypotheses = sorted(j for j, code in pairs.items() if code != SOURCE)
        true_supported = sorted(j for j in hypotheses if pairs[j] == SUPPORTED)
        scores = {j: float(np.dot(image_vectors[image_index], text_vectors[j])) for j in hypotheses}
        score_order = sorted(hypotheses, key=lambda j: (scores[j], j))
        score_bins = [list(map(int, part)) for part in np.array_split(np.asarray(score_order, dtype=np.int64), 2)]
        if control == "count_only":
            chosen = sorted(map(int, rng.choice(np.asarray(hypotheses, dtype=np.int64), size=len(true_supported), replace=False)))
        else:
            chosen = []
            for bucket in score_bins:
                count = sum(pairs[j] == SUPPORTED for j in bucket)
                chosen.extend(map(int, rng.choice(np.asarray(bucket, dtype=np.int64), size=count, replace=False)))
            chosen.sort()
        chosen_set, true_set = set(chosen), set(true_supported)
        overlap = len(chosen_set & true_set)
        bin_lookup = {j: b for b, bucket in enumerate(score_bins) for j in bucket}
        bin_records = [{"bin": b, "hypothesis_text_indices": bucket,
                        "supported_count": sum(pairs[j] == SUPPORTED for j in bucket),
                        "promoted_count": sum(j in chosen_set for j in bucket),
                        "quota_exchangeable": 0 < sum(pairs[j] == SUPPORTED for j in bucket) < len(bucket)}
                       for b, bucket in enumerate(score_bins)]
        sampling_buckets = [hypotheses] if control == "count_only" else score_bins
        sampling_units = [{"candidate_count": len(bucket), "supported_quota": sum(pairs[j] == SUPPORTED for j in bucket),
                           "exchangeable": 0 < sum(pairs[j] == SUPPORTED for j in bucket) < len(bucket),
                           "possible_subsets": math.comb(len(bucket), sum(pairs[j] == SUPPORTED for j in bucket))}
                          for bucket in sampling_buckets]
        true_strings = Counter(text_strings[j] for j in true_supported)
        chosen_strings = Counter(text_strings[j] for j in chosen)
        string_overlap = sum((true_strings & chosen_strings).values())
        string_multiplicity = Counter(text_strings[j] for j in hypotheses)
        score_groups = {}
        for j in score_order:
            score_groups.setdefault(scores[j], []).append(j)
        exact_ties = [{"frozen_cosine": score, "text_indices": group} for score, group in score_groups.items() if len(group) > 1]
        true_scores, chosen_scores = [scores[j] for j in true_supported], [scores[j] for j in chosen]
        pooled_true_scores.extend(true_scores)
        pooled_chosen_scores.extend(chosen_scores)
        records.append({
            "image_index": image_index, "image_id": image_ids[image_index],
            "source_text_indices": sources, "source_text_ids": [text_ids[j] for j in sources],
            "hypotheses": [{"text_index": j, "text_id": text_ids[j], "text": text_strings[j], "reference_relation": RELATION_NAMES[pairs[j]],
                            "frozen_cosine": scores[j], "score_bin": bin_lookup[j]} for j in hypotheses],
            "supported_text_indices": true_supported, "supported_text_ids": [text_ids[j] for j in true_supported],
            "promoted_text_indices": chosen, "promoted_text_ids": [text_ids[j] for j in chosen],
            "source_count": len(sources), "hypothesis_count": len(hypotheses), "promotion_count": len(chosen),
            "label_overlap_count": overlap, "changed_promotions": len(chosen) - overlap,
            "symmetric_difference_count": 2 * (len(chosen) - overlap),
            "exact_string_overlap_count": string_overlap,
            "exact_string_changed_promotions": len(chosen) - string_overlap,
            "exact_string_symmetric_difference_count": 2 * (len(chosen) - string_overlap),
            "exact_string_changed_image": true_strings != chosen_strings,
            "duplicate_hypothesis_string_groups": sum(count > 1 for count in string_multiplicity.values()),
            "duplicate_hypothesis_strings_beyond_first": sum(count - 1 for count in string_multiplicity.values()),
            "exact_score_tie_groups": exact_ties,
            "exact_score_tied_candidates": sum(len(group["text_indices"]) for group in exact_ties),
            "exact_score_ties_beyond_first": sum(len(group["text_indices"]) - 1 for group in exact_ties),
            "score_tie_crosses_bin_boundary": bool(score_bins[0] and score_bins[1] and scores[score_bins[0][-1]] == scores[score_bins[1][0]]),
            "sampling_units": sampling_units,
            "assignment_exchangeable": any(unit["exchangeable"] for unit in sampling_units),
            "possible_assignments": math.prod(unit["possible_subsets"] for unit in sampling_units),
            "source_target_share": len(sources) / (len(sources) + len(chosen)),
            "promoted_reference_label_counts": {RELATION_NAMES[code]: sum(pairs[j] == code for j in chosen)
                                               for code in (SUPPORTED, CONTRADICTED, NEUTRAL)},
            "score_bins": bin_records, "frozen_score_diagnostics": _score_diagnostics(true_scores, chosen_scores),
        })
    source_count = sum(row["source_count"] for row in records)
    promotion_count = sum(row["promotion_count"] for row in records)
    overlap_count = sum(row["label_overlap_count"] for row in records)
    result = {
        "schema_version": 1, "control": control, "assignment_seed": assignment_seed,
        "provenance": dict(provenance), "source_sha256": _file_sha256(Path(__file__)),
        "random_generator": "numpy.random.Generator(PCG64)", "numpy_version": np.__version__,
        "image_iteration_order": "ascending global manifest image index",
        "sampling_rule": "uniform subsets without replacement; one fixed assignment across all optimizer seeds, learning rates and epochs",
        "stratification": {"bins": 2, "score": "float64 cosine of explicitly L2-normalized cached frozen features",
                            "order": "ascending cosine, then ascending global manifest text index",
                            "odd_count_rule": "lower-score bin receives the extra candidate"},
        "global_image_count": len(image_ids), "global_text_count": len(text_ids),
        "training_image_indices": indices, "records": records,
        "summary": {
            "training_images": len(records), "hypothesis_candidates": sum(row["hypothesis_count"] for row in records),
            "source_positives": source_count, "promoted_hypotheses": promotion_count,
            "label_overlap_count": overlap_count,
            "label_overlap_fraction": overlap_count / promotion_count if promotion_count else None,
            "changed_promotions": promotion_count - overlap_count,
            "symmetric_difference_count": 2 * (promotion_count - overlap_count),
            "images_with_changed_assignment": sum(row["changed_promotions"] > 0 for row in records),
            "exact_string_overlap_count": sum(row["exact_string_overlap_count"] for row in records),
            "exact_string_changed_promotions": sum(row["exact_string_changed_promotions"] for row in records),
            "images_with_changed_exact_string_assignment": sum(row["exact_string_changed_image"] for row in records),
            "images_with_duplicate_hypothesis_strings": sum(row["duplicate_hypothesis_string_groups"] > 0 for row in records),
            "duplicate_hypothesis_string_groups": sum(row["duplicate_hypothesis_string_groups"] for row in records),
            "duplicate_hypothesis_strings_beyond_first": sum(row["duplicate_hypothesis_strings_beyond_first"] for row in records),
            "images_with_exact_score_ties": sum(bool(row["exact_score_tie_groups"]) for row in records),
            "exact_score_tie_groups": sum(len(row["exact_score_tie_groups"]) for row in records),
            "exact_score_ties_beyond_first": sum(row["exact_score_ties_beyond_first"] for row in records),
            "images_with_score_tie_across_bin_boundary": sum(row["score_tie_crosses_bin_boundary"] for row in records),
            "exchangeable_images": sum(row["assignment_exchangeable"] for row in records),
            "forced_images": sum(not row["assignment_exchangeable"] for row in records),
            "exchangeable_sampling_units": sum(unit["exchangeable"] for row in records for unit in row["sampling_units"]),
            "forced_sampling_units": sum(not unit["exchangeable"] for row in records for unit in row["sampling_units"]),
            "exchangeable_score_bins": sum(bucket["quota_exchangeable"] for row in records for bucket in row["score_bins"]),
            "forced_score_bins": sum(not bucket["quota_exchangeable"] for row in records for bucket in row["score_bins"]),
            "one_image_ownership_verified": True,
            "eligible_reverse_anchors_source": source_count,
            "eligible_reverse_anchors_supported": source_count + promotion_count,
            "eligible_reverse_anchors_control": source_count + promotion_count,
            "reverse_anchor_counts_preserved_for_any_image_batch": True,
            "frozen_score_diagnostics": _score_diagnostics(pooled_true_scores, pooled_chosen_scores),
            "mean_image_absolute_score_difference": float(np.mean([row["frozen_score_diagnostics"]["mean_absolute_sorted_difference"]
                                                                    for row in records if row["promotion_count"]])) if promotion_count else None,
        },
    }
    result["assignment_content_sha256"] = assignment_content_sha256(result)
    validate_assignment(result)
    return result


def validate_assignment(assignment: Mapping[str, Any], expected_provenance: Mapping[str, str] | None = None,
                        expected_control: str | None = None, expected_seed: int | None = None,
                        verify_source: bool = True) -> None:
    """Check content integrity and the invariants needed by batch application."""
    if assignment.get("schema_version") != 1 or assignment.get("control") not in CONTROLS:
        raise ValueError("Unsupported promotion-assignment schema or control")
    if assignment.get("assignment_content_sha256") != assignment_content_sha256(assignment):
        raise ValueError("Promotion-assignment canonical content hash mismatch")
    _check_provenance(assignment["provenance"])
    if expected_provenance is not None:
        for name, value in expected_provenance.items():
            if assignment["provenance"].get(name) != value:
                raise ValueError(f"Promotion assignment provenance mismatch for {name}")
    if expected_control is not None and assignment["control"] != expected_control:
        raise ValueError("Promotion assignment control mismatch")
    if expected_seed is not None and assignment["assignment_seed"] != expected_seed:
        raise ValueError("Promotion assignment seed mismatch")
    if verify_source and assignment["source_sha256"] != _file_sha256(Path(__file__)):
        raise ValueError("Promotion assignment was created by different module source")
    records = assignment["records"]
    image_indices = [row["image_index"] for row in records]
    if image_indices != assignment["training_image_indices"] or image_indices != sorted(set(image_indices)):
        raise ValueError("Assignment image records must follow unique manifest order")
    owners, sources_total, promotions_total, overlap_total = set(), 0, 0, 0
    for row in records:
        source = set(row["source_text_indices"])
        hypothesis_records = row["hypotheses"]
        hypotheses = {hyp["text_index"] for hyp in hypothesis_records}
        supported = {hyp["text_index"] for hyp in hypothesis_records if hyp["reference_relation"] == "supported"}
        chosen = set(row["promoted_text_indices"])
        if not source or source & hypotheses or len(hypotheses) != len(hypothesis_records) or owners & (source | hypotheses):
            raise ValueError("Assignment candidate ownership is invalid")
        owners.update(source | hypotheses)
        if not chosen <= hypotheses or supported != set(row["supported_text_indices"]) or len(chosen) != len(supported):
            raise ValueError("Assignment promotion counts or candidates are invalid")
        if len(chosen) != len(row["promoted_text_indices"]) or len(chosen) != row["promotion_count"]:
            raise ValueError("Assignment promotions must be unique and count matched")
        if len(source) != row["source_count"] or len(hypotheses) != row["hypothesis_count"]:
            raise ValueError("Assignment candidate count diagnostics are inconsistent")
        overlap = len(chosen & supported)
        if overlap != row["label_overlap_count"] or len(chosen) - overlap != row["changed_promotions"]:
            raise ValueError("Assignment label-overlap diagnostics are inconsistent")
        strings = {hyp["text_index"]: hyp["text"] for hyp in hypothesis_records}
        exact_overlap = sum((Counter(strings[j] for j in supported) & Counter(strings[j] for j in chosen)).values())
        if row["exact_string_overlap_count"] != exact_overlap or row["exact_string_changed_promotions"] != len(chosen) - exact_overlap:
            raise ValueError("Assignment exact-string diagnostics are inconsistent")
        if len(row["score_bins"]) != 2:
            raise ValueError("Assignment must contain exactly two score bins")
        scores = {hyp["text_index"]: hyp["frozen_cosine"] for hyp in hypothesis_records}
        expected_bins = [list(map(int, part)) for part in np.array_split(np.asarray(sorted(hypotheses, key=lambda j: (scores[j], j)), dtype=np.int64), 2)]
        for b, bucket in enumerate(row["score_bins"]):
            if bucket["bin"] != b or bucket["hypothesis_text_indices"] != expected_bins[b]:
                raise ValueError("Assignment score bins disagree with frozen-score ordering")
            actual_supported = len(set(expected_bins[b]) & supported)
            actual_chosen = len(set(expected_bins[b]) & chosen)
            if bucket["supported_count"] != actual_supported or bucket["promoted_count"] != actual_chosen:
                raise ValueError("Assignment score-bin counts are inconsistent")
            if assignment["control"] == "score_stratified" and actual_chosen != actual_supported:
                raise ValueError("Stratified assignment changes a score-bin promotion count")
        sources_total += len(source)
        promotions_total += len(chosen)
        overlap_total += overlap
    summary = assignment["summary"]
    if summary["source_positives"] != sources_total or summary["promoted_hypotheses"] != promotions_total or summary["label_overlap_count"] != overlap_total:
        raise ValueError("Assignment summary counts are inconsistent")
    if summary["eligible_reverse_anchors_supported"] != sources_total + promotions_total or summary["eligible_reverse_anchors_control"] != sources_total + promotions_total:
        raise ValueError("Assignment reverse-anchor counts are inconsistent")


def save_assignment(path: str | Path, assignment: Mapping[str, Any]) -> str:
    """Atomically create a JSON assignment; never replace a different assignment."""
    validate_assignment(assignment)
    path = Path(path)
    payload = json.dumps(assignment, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _canonical_json(json.loads(path.read_text())) != _canonical_json(assignment):
            raise ValueError(f"Refusing to replace a different frozen assignment: {path}")
        return _file_sha256(path)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".assignment-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic hard-link creation fails rather than overwriting a racing writer.
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if _canonical_json(json.loads(path.read_text())) != _canonical_json(assignment):
                raise ValueError(f"Refusing to replace a different frozen assignment: {path}")
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return _file_sha256(path)


def load_assignment(path: str | Path, *, expected_file_sha256: str | None = None,
                    expected_provenance: Mapping[str, str] | None = None,
                    expected_control: str | None = None, expected_seed: int | None = None) -> dict[str, Any]:
    path = Path(path)
    if expected_file_sha256 is not None and _file_sha256(path) != expected_file_sha256:
        raise ValueError("Promotion-assignment file hash mismatch")
    assignment = json.loads(path.read_text())
    validate_assignment(assignment, expected_provenance, expected_control, expected_seed)
    return assignment


def batch_positive_mask(relations: Tensor, image_indices: Sequence[int], text_indices: Sequence[int],
                        assignment: Mapping[str, Any]) -> Tensor:
    """Map a validated frozen assignment onto an unchanged canonical batch.

    ``image_indices`` and ``text_indices`` are global manifest indices in their
    exact batch order. Call ``load_assignment`` once before the training loop;
    expensive whole-record hash checks are intentionally not repeated per batch.
    Every own candidate must remain in the batch, not only promoted candidates.
    """
    image_indices, text_indices = list(map(int, image_indices)), list(map(int, text_indices))
    if relations.ndim != 2 or tuple(relations.shape) != (len(image_indices), len(text_indices)):
        raise ValueError("Relations must match batch image/text dimensions")
    if len(set(image_indices)) != len(image_indices) or len(set(text_indices)) != len(text_indices):
        raise ValueError("Batch image and text indices must be unique")
    if relations.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Relations must be integer encoded")
    if bool(((relations < 0) | (relations > 4)).any()):
        raise ValueError("Invalid canonical relation code")
    records = {int(row["image_index"]): row for row in assignment["records"]}
    positions = {index: position for position, index in enumerate(text_indices)}
    mask = relations == SOURCE
    for row_index, image_index in enumerate(image_indices):
        if image_index not in records:
            raise ValueError("Batch includes an image absent from the frozen training assignment")
        record = records[image_index]
        own = set(record["source_text_indices"]) | {entry["text_index"] for entry in record["hypotheses"]}
        if not own <= positions.keys():
            raise ValueError("Canonical batch omitted an own source or hypothesis candidate")
        actual_sources = {text_indices[j] for j in (relations[row_index] == SOURCE).nonzero(as_tuple=False).flatten().tolist()}
        actual_supported = {text_indices[j] for j in (relations[row_index] == SUPPORTED).nonzero(as_tuple=False).flatten().tolist()}
        if actual_sources != set(record["source_text_indices"]) or actual_supported != set(record["supported_text_indices"]):
            raise ValueError("Batch source/support labels disagree with the frozen assignment inputs")
        promoted = [positions[j] for j in record["promoted_text_indices"]]
        if promoted:
            mask[row_index, promoted] = True
    return mask


def apply_promotion_assignment(relations: Tensor, image_indices: Sequence[int], text_indices: Sequence[int],
                               assignment: Mapping[str, Any]) -> Tensor:
    """Return same-shape relations: source=1, assigned hypothesis positive=2, else=0.

    Use the returned matrix with the unchanged ``multipositive`` loss. It has
    unit weights, all original source captions, exactly matched per-image
    positive counts, and the same candidate matrix. Original relations are not
    modified. Reference contradiction/neutral labels are preserved only in the
    immutable assignment record, not used as weights in this control.
    """
    mask = batch_positive_mask(relations, image_indices, text_indices, assignment)
    output = torch.zeros_like(relations)
    output[mask] = SUPPORTED
    output[relations == SOURCE] = SOURCE
    return output
