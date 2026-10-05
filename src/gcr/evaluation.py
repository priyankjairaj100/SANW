"""Deterministic held-out metrics and paired image-cluster inference.

All inputs are newly encoded/adapted features. No historical aggregate is used.
Similarity uses float64 accumulation; retrieval ties use manifest candidate order.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

RETRIEVAL_TIE_POLICY = "descending float64 dot product, then ascending candidate manifest index"
PAIR_TIE_POLICY = "relation ranking: half credit; SugarCrepe/SugarCrepe++: zero credit"


def _features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite two-dimensional matrix")
    return values


def _id_map(ids: Sequence[str], name: str) -> dict[str, int]:
    result = {str(value): i for i, value in enumerate(ids)}
    if len(result) != len(ids):
        raise ValueError(f"Duplicate {name} IDs")
    return result


def supported_contradicted_accuracy(
    supported_scores: Sequence[float], contradicted_scores: Sequence[float]
) -> float:
    """Mean of every supported-by-contradicted pair, with half credit for ties."""
    positive = np.asarray(supported_scores, dtype=np.float64)
    negative = np.asarray(contradicted_scores, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both a supported and a contradicted caption are required")
    difference = positive[:, None] - negative[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def evaluate_relations(
    image_features: np.ndarray,
    text_features: np.ndarray,
    image_ids: Sequence[str],
    text_ids: Sequence[str],
    pairs: Iterable[dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Save each hypothesis score and each image's all-pair accuracy.

    ``image_ids`` must already select the held-out split. Images with insufficient
    labels are counted explicitly and are not silently given a score.
    """
    images, texts = _features(image_features), _features(text_features)
    image_lookup, text_lookup = _id_map(image_ids, "image"), _id_map(text_ids, "text")
    by_image: dict[str, dict[str, list[float]]] = {
        str(i): {"supported": [], "contradicted": []} for i in image_ids
    }
    raw_image, raw_text, raw_relation, raw_score = [], [], [], []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        iid, tid, relation = str(pair["image_id"]), str(pair["text_id"]), pair["relation"]
        if iid not in image_lookup or relation not in ("supported", "contradicted"):
            continue
        key = (iid, tid, relation)
        if key in seen:
            raise ValueError(f"Duplicated relation pair: {key}")
        seen.add(key)
        score = float(images[image_lookup[iid]] @ texts[text_lookup[tid]])
        by_image[iid][relation].append(score)
        raw_image.append(iid)
        raw_text.append(tid)
        raw_relation.append(relation)
        raw_score.append(score)
    eligible_ids, accuracy, pair_count, excluded = [], [], [], []
    for iid in map(str, image_ids):
        positives, negatives = by_image[iid]["supported"], by_image[iid]["contradicted"]
        if not positives or not negatives:
            excluded.append({"image_id": iid, "supported": len(positives), "contradicted": len(negatives)})
            continue
        eligible_ids.append(iid)
        accuracy.append(supported_contradicted_accuracy(positives, negatives))
        pair_count.append(len(positives) * len(negatives))
    if not accuracy:
        raise ValueError("No image has both supported and contradicted hypotheses")
    predictions = {
        "image_ids": np.asarray(eligible_ids, dtype=str),
        "image_accuracy": np.asarray(accuracy, dtype=np.float64),
        "comparison_counts": np.asarray(pair_count, dtype=np.int64),
        "pair_image_ids": np.asarray(raw_image, dtype=str),
        "pair_text_ids": np.asarray(raw_text, dtype=str),
        "pair_relations": np.asarray(raw_relation, dtype=str),
        "pair_scores": np.asarray(raw_score, dtype=np.float64),
    }
    metrics = {
        "accuracy": float(np.mean(accuracy)), "eligible_images": len(eligible_ids),
        "manifest_images": len(image_ids), "comparison_count": int(sum(pair_count)),
        "excluded_images": excluded,
    }
    return metrics, predictions


def evaluate_triplets(
    image_features: np.ndarray,
    text_features: np.ndarray,
    image_ids: Sequence[str],
    text_ids: Sequence[str],
    triplets: Sequence[dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    images, texts = _features(image_features), _features(text_features)
    image_lookup, text_lookup = _id_map(image_ids, "image"), _id_map(text_ids, "text")
    if not triplets:
        raise ValueError("No benchmark triplets")
    has_second = ["positive2_id" in row for row in triplets]
    if any(has_second) and not all(has_second):
        raise ValueError("Mixed single-positive and two-positive triplets")
    iid = np.asarray([str(row["image_id"]) for row in triplets], dtype=str)
    ids = np.asarray([str(row["id"]) for row in triplets], dtype=str)
    _id_map(ids, "triplet")
    selected_images = images[[image_lookup[i] for i in iid]]
    predictions = {"item_ids": ids, "image_ids": iid,
                   "categories": np.asarray([str(row["category"]) for row in triplets], dtype=str)}
    for field in ("positive1_id", "negative_id") + (("positive2_id",) if all(has_second) else ()):
        text_id_values = np.asarray([str(row[field]) for row in triplets], dtype=str)
        selected_texts = texts[[text_lookup[i] for i in text_id_values]]
        predictions[field.replace("_id", "_ids")] = text_id_values
        predictions[field.replace("_id", "_scores")] = np.einsum("ij,ij->i", selected_images, selected_texts)
    p1 = predictions["positive1_scores"] > predictions["negative_scores"]
    predictions["positive1_correct"] = p1
    metrics = {"items": len(triplets), "images": len(np.unique(iid)), "positive1_accuracy": float(p1.mean())}
    if all(has_second):
        p2 = predictions["positive2_scores"] > predictions["negative_scores"]
        predictions["positive2_correct"] = p2
        predictions["correct"] = p1 & p2
        metrics["positive2_accuracy"] = float(p2.mean())
        metrics["both_accuracy"] = float((p1 & p2).mean())
    else:
        predictions["correct"] = p1
    metrics["accuracy"] = float(predictions["correct"].mean())
    return metrics, predictions


def ranks_to_metrics(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0 or np.any(ranks < 1):
        raise ValueError("Ranks must be a nonempty vector of one-based ranks")
    return {"r1": float(np.mean(ranks <= 1)), "r5": float(np.mean(ranks <= 5)),
            "r10": float(np.mean(ranks <= 10)), "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks))}


def full_pool_ranks(
    query_features: np.ndarray,
    candidate_features: np.ndarray,
    relevant_indices: Sequence[Sequence[int]],
    block_size: int = 128,
    top_k: int = 10,
) -> dict[str, np.ndarray]:
    """Exact full-pool retrieval; retain ranks and deterministic top-k predictions.

    A query's rank is its first relevant candidate under stable score ordering.
    Memory is O(block_size * number_candidates), with no subsampling.
    """
    queries, candidates = _features(query_features), _features(candidate_features)
    if queries.shape[1] != candidates.shape[1] or len(relevant_indices) != len(queries):
        raise ValueError("Feature shapes or relevance rows do not match")
    if block_size < 1 or top_k < 1 or len(candidates) < 1:
        raise ValueError("Block size, top_k and candidate count must be positive")
    k = min(top_k, len(candidates))
    candidate_indices = np.arange(len(candidates))
    ranks = np.empty(len(queries), dtype=np.int64)
    correct_indices = np.empty(len(queries), dtype=np.int64)
    correct_scores = np.empty(len(queries), dtype=np.float64)
    top_indices = np.empty((len(queries), k), dtype=np.int64)
    top_scores = np.empty((len(queries), k), dtype=np.float64)
    for start in range(0, len(queries), block_size):
        scores = queries[start:start + block_size] @ candidates.T
        for offset, row in enumerate(scores):
            query_index = start + offset
            relevant = np.unique(np.asarray(relevant_indices[query_index], dtype=np.int64))
            if relevant.size == 0 or relevant.min() < 0 or relevant.max() >= len(candidates):
                raise ValueError(f"Query {query_index} has empty or invalid relevance")
            best = int(relevant[np.argmax(row[relevant])])
            value = row[best]
            ranks[query_index] = 1 + np.count_nonzero(row > value) + np.count_nonzero((row == value) & (candidate_indices < best))
            correct_indices[query_index], correct_scores[query_index] = best, value
            # Include every kth-boundary tie before deterministic lexicographic selection.
            threshold = np.partition(row, len(row) - k)[len(row) - k]
            shortlist = np.flatnonzero(row >= threshold)
            order = np.lexsort((shortlist, -row[shortlist]))[:k]
            chosen = shortlist[order]
            top_indices[query_index], top_scores[query_index] = chosen, row[chosen]
    return {"ranks": ranks, "best_relevant_indices": correct_indices,
            "best_relevant_scores": correct_scores, "top_indices": top_indices, "top_scores": top_scores}


def evaluate_retrieval(
    image_features: np.ndarray,
    text_features: np.ndarray,
    image_ids: Sequence[str],
    text_ids: Sequence[str],
    pairs: Sequence[dict],
    block_size: int = 128,
) -> tuple[dict, dict[str, np.ndarray]]:
    image_lookup, text_lookup = _id_map(image_ids, "image"), _id_map(text_ids, "text")
    image_relevance: list[list[int]] = [[] for _ in image_ids]
    text_relevance: list[list[int]] = [[] for _ in text_ids]
    for pair in pairs:
        if pair["relation"] != "source":
            continue
        ii, ti = image_lookup[str(pair["image_id"])], text_lookup[str(pair["text_id"])]
        image_relevance[ii].append(ti)
        text_relevance[ti].append(ii)
    i2t = full_pool_ranks(image_features, text_features, image_relevance, block_size)
    t2i = full_pool_ranks(text_features, image_features, text_relevance, block_size)
    predictions = {"image_ids": np.asarray(image_ids, dtype=str), "text_ids": np.asarray(text_ids, dtype=str),
                   "text_source_image_ids": np.asarray([str(image_ids[row[0]]) for row in text_relevance], dtype=str)}
    for prefix, values in (("i2t", i2t), ("t2i", t2i)):
        predictions.update({f"{prefix}_{key}": value for key, value in values.items()})
    metrics = {"images": len(image_ids), "texts": len(text_ids), "i2t": ranks_to_metrics(i2t["ranks"]),
               "t2i": ranks_to_metrics(t2i["ranks"]), "tie_policy": RETRIEVAL_TIE_POLICY}
    return metrics, predictions


def paired_image_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    image_ids: Sequence[str],
    *,
    replicates: int = 10000,
    seed: int = 20261003,
    family_size: int = 6,
    alpha: float = 0.05,
    batch_size: int = 250,
) -> dict:
    """Average selected seeds, then resample paired image clusters.

    Inputs have shape [seed, item]. For image-averaged relation accuracy each
    image is one item. For caption benchmarks all items remain in their image
    cluster; each replicate recomputes the item-weighted mean. Intervals are
    conditional on the selected checkpoints and do not resample training seeds.
    """
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(image_ids):
        raise ValueError("Paired predictions must match [seed,item] and image IDs")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.size == 0:
        raise ValueError("Predictions must be finite and nonempty")
    if replicates < 1 or family_size < 1 or not 0 < alpha < 1 or batch_size < 1:
        raise ValueError("Invalid bootstrap parameters")
    unique, inverse = np.unique(np.asarray(image_ids, dtype=str), return_inverse=True)
    difference = (a - b).mean(axis=0)
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        draws = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        samples[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = alpha / (2 * family_size)
    lower, upper = np.quantile(samples, [tail, 1 - tail], method="linear")
    by_seed = (a - b).mean(axis=1)
    return {"difference": float(difference.mean()), "ci_lower": float(lower), "ci_upper": float(upper),
            "confidence": 1 - alpha / family_size, "family_size": family_size,
            "replicates": replicates, "bootstrap_seed": seed, "images": len(unique), "items": len(image_ids),
            "training_seeds": len(a), "seed_differences": by_seed.tolist(),
            "seed_difference_std": float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else None,
            "conditioning": "selected training runs; seed predictions averaged before image-cluster resampling",
            "quantile_method": "linear", "bootstrap_differences": samples}

