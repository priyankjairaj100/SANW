"""Development-only evaluation of one fixed constrained bilinear state.

There is no alpha search, checkpoint selection, model fitting, or test loader.
The canonical pair score has fixed float64 reductions. GEMM only screens
possible retrieval winners; every candidate within a conservative numerical
error band is rescored by that same pair function.
"""
from __future__ import annotations

from itertools import combinations
import re
import unicodedata

import numpy as np


BOOTSTRAP_REPLICATES = 100000
BOOTSTRAP_SEED = 20261007
FAMILY_SIZE = 80
TIE_POLICY = "descending canonical float64 score then ascending gallery manifest index"


def _finite_matrix(value, name):
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    return result


def canonical_project(values, matrix, mean=None):
    """Unoptimized einsum projection, independent of input row batching."""
    values, matrix = _finite_matrix(values, "values"), _finite_matrix(matrix, "matrix")
    if values.shape[1] != matrix.shape[0]:
        raise ValueError("Projection dimension mismatch")
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float64)
        if mean.shape != (values.shape[1],) or not np.isfinite(mean).all():
            raise ValueError("Invalid projection mean")
        values = np.ascontiguousarray(values - mean)
    return np.einsum("nd,dr->nr", values, matrix, optimize=False)


class CanonicalScorer:
    """One pair function for every query direction and composition endpoint."""

    def __init__(self, mu_v, mu_t, U, V, A):
        self.mu_v, self.mu_t = map(lambda x: np.asarray(x, dtype=np.float64), (mu_v, mu_t))
        self.U, self.V, self.A = (_finite_matrix(x, name) for x, name in ((U, "U"), (V, "V"), (A, "A")))
        if self.U.shape != self.V.shape or self.A.shape != (self.U.shape[1], self.U.shape[1]):
            raise ValueError("Bilinear state dimensions disagree")
        if self.mu_v.shape != (self.U.shape[0],) or self.mu_t.shape != self.mu_v.shape:
            raise ValueError("Bilinear means have the wrong dimension")
        if not np.isfinite(self.mu_v).all() or not np.isfinite(self.mu_t).all():
            raise ValueError("Bilinear means must be finite")

    def prepare(self, images, texts):
        images, texts = _finite_matrix(images, "images"), _finite_matrix(texts, "texts")
        left = canonical_project(canonical_project(images, self.U, self.mu_v), self.A)
        right = canonical_project(texts, self.V, self.mu_t)
        return images, texts, left, right

    def pair_scores(self, images, texts):
        return canonical_pair_scores(*self.prepare(images, texts))


def canonical_pair_scores(images, texts, left, right):
    """Fixed pair reductions; callers supply aligned prepared representations."""
    images, texts, left, right = (np.asarray(x, dtype=np.float64) for x in (images, texts, left, right))
    if images.shape != texts.shape or left.shape != right.shape or len(images) != len(left):
        raise ValueError("Canonical pairs must be aligned")
    base = np.sum(np.ascontiguousarray(images * texts), axis=1, dtype=np.float64)
    residual = np.sum(np.ascontiguousarray(left * right), axis=1, dtype=np.float64)
    result = base + residual
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite canonical score")
    return result


def exact_retrieval(images, texts, owner, scorer=None, *, query_block=64):
    """Exhaustive-gallery top-1 with canonical rescoring of all possible winners.

    The norm bound covers both GEMM and pairwise dot-product roundoff, plus
    addition. The factor 32 exceeds the standard gamma_d error bound here.
    It is a numerical screen only, not a fixed top-k or score budget.
    """
    images, texts = _finite_matrix(images, "images"), _finite_matrix(texts, "texts")
    if not len(images) or not len(texts) or images.shape[1] != texts.shape[1]:
        raise ValueError("Require nonempty aligned feature dimensions")
    owner = np.asarray(owner, dtype=np.int64)
    if owner.shape != (len(texts),) or np.any(owner < 0) or np.any(owner >= len(images)):
        raise ValueError("Invalid unique source ownership vector")
    if not isinstance(query_block, int) or query_block <= 0:
        raise ValueError("query_block must be a positive integer")
    if scorer is None:
        left, right = np.zeros((len(images), 1)), np.zeros((len(texts), 1))
    else:
        images, texts, left, right = scorer.prepare(images, texts)
    dimension, rank = images.shape[1], left.shape[1]
    if (dimension + rank + 4) * np.finfo(np.float64).eps >= 0.001:
        raise ValueError("Dimensions exceed the conservative roundoff-bound scope")
    raw = {"owner": owner}
    for direction, queries, gallery, qlow, glow in (
        ("i2t", images, texts, left, right), ("t2i", texts, images, right, left)
    ):
        winners = np.empty(len(queries), dtype=np.int64)
        win_scores = np.empty(len(queries), dtype=np.float64)
        candidate_counts = np.empty(len(queries), dtype=np.int64)
        bounds = np.empty(len(queries), dtype=np.float64)
        saved_q, saved_g, saved_s = [], [], []
        gallery_norm = float(np.linalg.norm(gallery, axis=1).max())
        gallery_low_norm = float(np.linalg.norm(glow, axis=1).max())
        for start in range(0, len(queries), query_block):
            q, ql = queries[start:start + query_block], qlow[start:start + query_block]
            approximate = q @ gallery.T + ql @ glow.T
            guard = (32 * (dimension + rank + 4) * np.finfo(np.float64).eps
                     * (1 + np.linalg.norm(q, axis=1) * gallery_norm
                        + np.linalg.norm(ql, axis=1) * gallery_low_norm))
            if not np.isfinite(approximate).all() or not np.isfinite(guard).all():
                raise ValueError("Nonfinite full-gallery screening score or bound")
            for row in range(len(q)):
                selected = np.flatnonzero(approximate[row] >= approximate[row].max() - 2 * guard[row])
                qi = np.full(len(selected), start + row, dtype=np.int64)
                ii, tt = (qi, selected) if direction == "i2t" else (selected, qi)
                scores = canonical_pair_scores(images[ii], texts[tt], left[ii], right[tt])
                # Ascending selected indices plus first-maximum tie breaking.
                best = int(np.argmax(scores))
                winners[start + row], win_scores[start + row] = selected[best], scores[best]
                candidate_counts[start + row], bounds[start + row] = len(selected), guard[row]
                saved_q.append(qi); saved_g.append(selected); saved_s.append(scores)
        correct = owner[winners] == np.arange(len(images)) if direction == "i2t" else winners == owner
        raw.update({f"{direction}_top_indices": winners, f"{direction}_top_scores": win_scores,
                    f"{direction}_correct": correct, f"{direction}_candidate_counts": candidate_counts,
                    f"{direction}_roundoff_bounds": bounds,
                    f"{direction}_rescored_query_indices": np.concatenate(saved_q),
                    f"{direction}_rescored_gallery_indices": np.concatenate(saved_g),
                    f"{direction}_rescored_scores": np.concatenate(saved_s)})
    summary = {"image_count": len(images), "text_count": len(texts),
               "i2t_r1": float(raw["i2t_correct"].mean()), "t2i_r1": float(raw["t2i_correct"].mean()),
               "tie_rule": TIE_POLICY, "relevance": "source_caption_ownership",
               "scoring": "one canonical pair function; exhaustive GEMM with conservative roundoff screening",
               "test_dependent_routing": False, "fixed_top_k_shortlist": False}
    return summary, raw


def caption_key(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def composition_metrics(data, scorer=None):
    """All original validation triples, plus distinct-source pair diagnostics.

    Original source/support/contradiction comparisons retain their original
    labels. The source-pair metric excludes exact normalized label conflicts,
    matching the documented v7 source-pair definition. No margin filter applies.
    """
    raw_i, raw_t, raw_label = [], [], []
    for image in data.split_indices["validation"]:
        for text in sorted(data.pairs[image]):
            raw_i.append(image); raw_t.append(text); raw_label.append(data.pairs[image][text])
    raw_i, raw_t, raw_label = (np.asarray(x, dtype=np.int64) for x in (raw_i, raw_t, raw_label))
    images, texts = np.asarray(data.images, dtype=np.float64), np.asarray(data.texts, dtype=np.float64)
    scores = (np.sum(images[raw_i] * texts[raw_t], axis=1, dtype=np.float64) if scorer is None
              else scorer.pair_scores(images[raw_i], texts[raw_t]))
    raw = {"raw_image_index": raw_i, "raw_text_index": raw_t, "raw_relation": raw_label, "raw_score": scores}
    original_ids, original_accuracy, original_margin, original_counts = [], [], [], []
    source_ids, source_accuracy, source_margin, source_counts = [], [], [], []
    excluded_conflicts = 0
    for image in data.split_indices["validation"]:
        mask = raw_i == image
        labels, values, indices = raw_label[mask], scores[mask], raw_t[mask]
        source, supported, contra = values[labels == 1], values[labels == 2], values[labels == 3]
        if min(len(source), len(supported), len(contra)):
            margin = np.minimum(source[:, None, None], supported[None, :, None]) - contra[None, None, :]
            original_ids.append(data.image_ids[image]); original_accuracy.append(float((margin > 0).mean()))
            original_margin.append(float(margin.mean())); original_counts.append(margin.size)
        keys = {j: caption_key(data.manifest["texts"][j]["text"]) for j in indices}
        positive_keys = {keys[j] for j, code in zip(indices, labels) if code in (1, 2)}
        lookup = dict(zip(indices.tolist(), values.tolist()))
        negatives = [j for j, code in zip(indices, labels) if code == 3 and keys[j] not in positive_keys]
        excluded_conflicts += sum(code == 3 and keys[j] in positive_keys for j, code in zip(indices, labels))
        pairs = [(j, k) for j, k in combinations(indices[labels == 1], 2) if keys[j] != keys[k]]
        if pairs and negatives:
            margin = np.asarray([min(lookup[j], lookup[k]) - lookup[n] for j, k in pairs for n in negatives])
            source_ids.append(data.image_ids[image]); source_accuracy.append(float((margin > 0).mean()))
            source_margin.append(float(margin.mean())); source_counts.append(margin.size)
    summary = {}
    for name, ids, accuracy, margin, counts in (
        ("original", original_ids, original_accuracy, original_margin, original_counts),
        ("source_pair", source_ids, source_accuracy, source_margin, source_counts)
    ):
        if not ids:
            raise ValueError(f"No eligible {name} validation comparisons")
        raw.update({f"{name}_image_ids": np.asarray(ids), f"{name}_joint_accuracy": np.asarray(accuracy),
                    f"{name}_joint_margin": np.asarray(margin), f"{name}_triplet_counts": np.asarray(counts)})
        summary[name] = {"image_count": len(ids), "joint_accuracy": float(np.mean(accuracy)),
                         "mean_joint_margin": float(np.mean(margin)), "triplet_count": int(sum(counts)),
                         "averaging": "equal_image_weight_then_all_declared_triplets",
                         "strict_positive_comparison": True, "training_eligibility_filter": False}
    summary["source_pair"]["excluded_conflicting_negatives"] = int(excluded_conflicts)
    summary["original"]["excluded_conflicting_negatives"] = 0
    summary["neutral_as_negative"] = False
    return summary, raw


def paired_cluster_bootstrap(delta, cluster_ids, *, replicates=BOOTSTRAP_REPLICATES,
                             seed=BOOTSTRAP_SEED, family_size=FAMILY_SIZE):
    """Image-cluster bootstrap, retaining every simulated value for auditing."""
    delta, cluster_ids = np.asarray(delta, dtype=np.float64), np.asarray(cluster_ids)
    if delta.ndim != 1 or cluster_ids.shape != delta.shape or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Require finite aligned paired changes and cluster IDs")
    if not isinstance(replicates, int) or replicates <= 0 or not isinstance(family_size, int) or family_size <= 0:
        raise ValueError("Positive integer bootstrap and family counts required")
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=delta), np.bincount(inverse)
    rng, samples = np.random.default_rng(seed), np.empty(replicates, dtype=np.float64)
    if np.all(counts == counts[0]):
        values, frequencies = np.unique(sums, return_counts=True)
        for lo in range(0, replicates, 4096):
            draws = rng.multinomial(len(sums), frequencies / len(sums), size=min(4096, replicates - lo))
            samples[lo:lo + len(draws)] = (draws @ values) / (len(sums) * counts[0])
        algorithm = "grouped_multinomial_exact_equal_cluster_distribution"
    else:
        for lo in range(0, replicates, 250):
            draws = rng.integers(0, len(sums), size=(min(250, replicates - lo), len(sums)))
            samples[lo:lo + len(draws)] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        algorithm = "direct_paired_image_cluster_ratio"
    tail = 0.05 / (2 * family_size)
    low, high = np.quantile(samples, [tail, 1 - tail], method="linear")
    summary = {"difference": float(delta.mean()), "ci_lower": float(low), "ci_upper": float(high),
               "replicates": replicates, "bootstrap_seed": seed, "family_size": family_size,
               "familywise_alpha": 0.05, "tail_probability": tail, "image_clusters": len(sums),
               "items": len(delta), "algorithm": algorithm,
               "conditioning": "one training-objective-selected state fixed; development image clusters resampled"}
    return summary, samples


def development_gate(frozen_composition, composition, retention, *, update_norm):
    """Evaluate the predeclared gate; never choose among trained candidates."""
    for direction in ("i2t", "t2i"):
        effect = retention[direction]
        if (effect["replicates"], effect["bootstrap_seed"], effect["family_size"]) != (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILY_SIZE):
            raise ValueError("Development bootstrap differs from the protocol")
        if not all(np.isfinite(effect[k]) for k in ("difference", "ci_lower", "ci_upper")):
            raise ValueError("Nonfinite uncertainty statistic")
    original_gain = composition["original"]["joint_accuracy"] - frozen_composition["original"]["joint_accuracy"]
    source_gain = composition["source_pair"]["joint_accuracy"] - frozen_composition["source_pair"]["joint_accuracy"]
    if not all(np.isfinite(x) for x in (original_gain, source_gain, update_norm)):
        raise ValueError("Nonfinite development statistic")
    checks = {"nonzero_trained_update": update_norm > 0, "original_joint_improvement": original_gain > 0,
              "source_pair_joint_non_decrease": source_gain >= 0}
    for direction in ("i2t", "t2i"):
        checks[f"{direction}_mean_non_decrease"] = retention[direction]["difference"] >= 0
        checks[f"{direction}_strict_adjusted_retention"] = retention[direction]["ci_lower"] > -0.01
    return {"passed": all(checks.values()), "checks": checks, "original_joint_gain": original_gain,
            "source_pair_joint_gain": source_gain, "checkpoint_search": False, "alpha_search": False,
            "status": "development_only_not_a_heldout_success"}
