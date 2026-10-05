"""Development-only source-pair metrics and globally scaled token selection.

This module never changes the held-out evaluator or its bootstrap algorithm.
The grouped multinomial sampler has the same cluster-bootstrap distribution
as drawing cluster IDs; it does not promise the same finite RNG realization.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from .practical_source_pair_v7 import caption_key


ALPHA_GRID = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
BOOTSTRAP_REPLICATES = 100000
BOOTSTRAP_SEED = 20261007
FAMILY_SIZE = 80


def calibrated_score(base, correction, alpha):
    """One alpha multiplies the bounded correction for every pair and task."""
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Require a positive global correction scale at most one")
    base, correction = np.asarray(base, dtype=np.float64), np.asarray(correction, dtype=np.float64)
    if base.shape != correction.shape or not np.isfinite(base).all() or not np.isfinite(correction).all():
        raise ValueError("Require aligned finite baseline and correction arrays")
    return base + float(alpha) * correction


def source_pair_validation_metrics(data, raw_image_indices, raw_text_indices, raw_scores):
    """All valid source/source/contradiction triples, with equal image weights.

    Distinct means different normalized caption-key sequences. Contradictions
    identical to any declared source or supported caption are excluded as label
    conflicts. There is NO frozen-margin, repairability, or training-eligibility
    filter here. Unsupported/neutral captions do not become negatives.
    """
    images = np.asarray(raw_image_indices, dtype=np.int64)
    texts = np.asarray(raw_text_indices, dtype=np.int64)
    scores = np.asarray(raw_scores, dtype=np.float64)
    if not (images.ndim == texts.ndim == scores.ndim == 1 and len(images) == len(texts) == len(scores)):
        raise ValueError("Require aligned one-dimensional development score arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Development scores must be finite")
    lookup = {}
    for image, text, score in zip(images.tolist(), texts.tolist(), scores.tolist()):
        if (image, text) in lookup:
            raise ValueError("Duplicate image/text score row")
        lookup[image, text] = score
    expected = {(i, j) for i in data.split_indices["validation"] for j in data.pairs[i]}
    if set(lookup) != expected:
        raise ValueError("Scores must contain exactly the full declared validation relation set")
    ids, accuracies, margins, counts, excluded, pair_counts = [], [], [], [], [], []
    for image in data.split_indices["validation"]:
        labels = data.pairs[image]
        keys = {j: caption_key(data.manifest["texts"][j]["text"]) for j in labels}
        source = sorted(j for j, code in labels.items() if code == 1)
        positive_keys = {keys[j] for j, code in labels.items() if code in (1, 2)}
        negative = sorted(j for j, code in labels.items() if code == 3 and keys[j] not in positive_keys)
        excluded.append(sum(code == 3 and keys[j] in positive_keys for j, code in labels.items()))
        pairs = [(left, right) for left, right in combinations(source, 2) if keys[left] != keys[right]]
        if not pairs or not negative:
            continue
        values = np.asarray([min(lookup[image, left], lookup[image, right]) - lookup[image, neg]
                             for left, right in pairs for neg in negative], dtype=np.float64)
        ids.append(data.image_ids[image]); accuracies.append(float((values > 0).mean()))
        margins.append(float(values.mean())); counts.append(len(values)); pair_counts.append(len(pairs))
    if not ids:
        raise ValueError("No valid source-pair validation triples")
    raw = {"image_ids": np.asarray(ids), "paired_joint_accuracy": np.asarray(accuracies),
           "paired_joint_margin": np.asarray(margins), "triplet_count": np.asarray(counts),
           "distinct_source_pair_count": np.asarray(pair_counts),
           "raw_image_index": images, "raw_text_index": texts, "raw_score": scores}
    summary = {"image_count": len(ids), "declared_validation_image_count": len(data.split_indices["validation"]),
               "paired_joint_accuracy": float(np.mean(accuracies)),
               "mean_paired_joint_margin": float(np.mean(margins)), "triplet_count": int(sum(counts)),
               "excluded_conflicting_negative_count": int(sum(excluded)),
               "averaging": "equal_image_weight_then_all_distinct_source_source_contradiction_triplets",
               "frozen_margin_filter": False, "training_eligibility_filter": False,
               "neutral_as_negative": False, "strict_positive_comparison": True}
    return summary, raw


def cluster_statistics(delta_by_item, cluster_ids):
    delta, ids = np.asarray(delta_by_item, dtype=np.float64), np.asarray(cluster_ids)
    if delta.ndim != 1 or ids.ndim != 1 or not len(delta) or len(delta) != len(ids) or not np.isfinite(delta).all():
        raise ValueError("Require one finite paired difference per clustered development query")
    _, inverse = np.unique(ids, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    return sums, counts


def grouped_bootstrap_moments(delta_by_item, cluster_ids):
    """Exact moments of the ordinary image-bootstrap mean, equal cluster sizes."""
    sums, counts = cluster_statistics(delta_by_item, cluster_ids)
    if not np.all(counts == counts[0]):
        raise ValueError("Analytic grouped moments require equal cluster sizes")
    means = sums / counts[0]
    return {"mean": float(means.mean()), "variance": float(means.var() / len(means)),
            "clusters": len(means), "items_per_cluster": int(counts[0])}


def development_bootstrap(delta_by_item, cluster_ids, *, replicates=BOOTSTRAP_REPLICATES,
                          seed=BOOTSTRAP_SEED, family_size=FAMILY_SIZE):
    """Paired image bootstrap; exact grouping when all query clusters are equal.

    For N image clusters with m queries each, the sample is sum(n_k*s_k)/(N*m),
    where n ~ Multinomial(N, frequency(s_k)/N). Unequal sizes use explicit image
    ID draws with the resampled query denominator, preserving ratio semantics.
    """
    if not isinstance(replicates, int) or replicates <= 0 or not isinstance(family_size, int) or family_size <= 0:
        raise ValueError("Require positive integer replicate and family counts")
    delta = np.asarray(delta_by_item, dtype=np.float64)
    sums, counts = cluster_statistics(delta, cluster_ids)
    rng, samples = np.random.default_rng(seed), np.empty(replicates, dtype=np.float64)
    equal = bool(np.all(counts == counts[0]))
    if equal:
        values, frequencies = np.unique(sums, return_counts=True)
        probabilities = frequencies.astype(np.float64) / len(sums)
        for start in range(0, replicates, 4096):
            draws = rng.multinomial(len(sums), probabilities, size=min(4096, replicates - start))
            samples[start:start + len(draws)] = (draws @ values) / (len(sums) * int(counts[0]))
        algorithm = "grouped_multinomial_equal_image_cluster_size"
    else:
        for start in range(0, replicates, 250):
            draws = rng.integers(0, len(sums), size=(min(250, replicates - start), len(sums)))
            samples[start:start + len(draws)] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        values = np.unique(sums)
        algorithm = "direct_paired_image_cluster_draws_unequal_sizes"
    tail = .05 / (2 * family_size)
    lower, upper = np.quantile(samples, [tail, 1 - tail], method="linear")
    summary = {"difference": float(delta.mean()), "ci_lower": float(lower), "ci_upper": float(upper),
               "family_size": family_size, "familywise_alpha": .05, "tail_probability": tail,
               "bootstrap_seed": int(seed), "replicates": replicates, "images": len(sums),
               "items": len(delta), "equal_cluster_sizes": equal,
               "items_per_cluster": int(counts[0]) if equal else None,
               "distinct_cluster_sums": len(values), "algorithm": algorithm,
               "conditioning": "one checkpoint and global alpha fixed; development image clusters resampled",
               "heldout_bootstrap_changed": False}
    return summary, samples


def select_calibrated_development(rows):
    """Select a nonzero epoch/alpha after the stronger two-direction dev gate.

    Rows contain epoch, alpha, update_norm, optimizer_steps, residual_rms,
    checkpoint, composition plus frozen_composition, and retention.i2t/t2i.
    Both retention entries are development_bootstrap summaries. No zero-epoch
    or alpha-zero fallback is allowed.
    """
    if len({(row["epoch"], row["alpha"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate development epoch/alpha candidate")
    eligible = []
    for row in rows:
        if row["alpha"] not in ALPHA_GRID:
            raise ValueError("Candidate alpha is outside the predeclared positive grid")
        values = [row["update_norm"], row["residual_rms"],
                  row["composition"]["paired_joint_accuracy"], row["composition"]["mean_paired_joint_margin"],
                  row["frozen_composition"]["paired_joint_accuracy"]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite candidate selection statistic")
        if row["epoch"] <= 0 or row["optimizer_steps"] <= 0 or row["update_norm"] <= 0 or row["residual_rms"] <= 1e-10:
            continue
        if row["composition"]["paired_joint_accuracy"] <= row["frozen_composition"]["paired_joint_accuracy"]:
            continue
        passing = True
        for direction in ("i2t", "t2i"):
            effect = row["retention"][direction]
            if effect["replicates"] != BOOTSTRAP_REPLICATES or effect["family_size"] != FAMILY_SIZE or effect["bootstrap_seed"] != BOOTSTRAP_SEED:
                raise ValueError("Selection uncertainty differs from the declared bootstrap")
            if not math.isfinite(effect["difference"]) or not math.isfinite(effect["ci_lower"]):
                raise ValueError("Nonfinite retention statistic")
            passing &= effect["difference"] >= 0 and effect["ci_lower"] > -0.01
        if passing:
            eligible.append(row)
    if not eligible:
        return {"selected_epoch": None, "selected_alpha": None, "eligible_count": 0,
                "status": "no_positive_gain_nonzero_epoch_alpha_passed_development_retention"}
    best = max(eligible, key=lambda row: (
        row["composition"]["paired_joint_accuracy"] - row["frozen_composition"]["paired_joint_accuracy"],
        row["composition"]["mean_paired_joint_margin"], -row["epoch"], -row["alpha"]))
    return {"selected_epoch": best["epoch"], "selected_alpha": best["alpha"], "eligible_count": len(eligible),
            "status": "development_selected_not_tested", "checkpoint": best["checkpoint"],
            "paired_joint_accuracy_gain": best["composition"]["paired_joint_accuracy"] - best["frozen_composition"]["paired_joint_accuracy"],
            "retention": best["retention"], "global_alpha_policy": "same scalar for every pair, task and gallery"}
