"""Common original-training holdout metrics for joint and LABCLIP candidates.

Gallery: all 1,200 original training images and 6,000 owned source captions.
Queries: the fixed 240 inner holdout owners and their 1,200 source captions.
The official development split and benchmark cases never contribute metrics.
"""
from __future__ import annotations

from types import SimpleNamespace
from fractions import Fraction
import math

import numpy as np

from .practical_constrained_evaluation_v8 import composition_metrics, exact_retrieval, paired_cluster_bootstrap


METRICS = ("i2t", "t2i", "original", "source_pair")
ENCODERS = ("vit_b32", "rn50")


class LABCLIPScorer:
    """Pure normalized full-rank transform; no frozen reference correction."""
    def __init__(self, weight):
        self.weight = np.asarray(weight, dtype=np.float64)
        if self.weight.ndim != 2 or self.weight.shape[0] != self.weight.shape[1] or not np.isfinite(self.weight).all():
            raise ValueError("Require a finite square LABCLIP weight")

    def transform(self, texts):
        texts = np.asarray(texts, dtype=np.float64)
        transformed = np.einsum("nd,kd->nk", texts, self.weight, optimize=False)
        norms = np.sqrt(np.sum(transformed * transformed, axis=1, keepdims=True, dtype=np.float64))
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("LABCLIP produced an invalid transformed text norm")
        return transformed / norms

    def prepare(self, images, texts):
        images = np.asarray(images, dtype=np.float64)
        transformed = self.transform(texts)
        return images, transformed, np.zeros((len(images), 1)), np.zeros((len(texts), 1))

    def pair_scores(self, images, texts):
        return np.sum(np.asarray(images, dtype=np.float64) * self.transform(texts), axis=1, dtype=np.float64)


def make_inner_view(data, split):
    """Inputs are already normalized once, with all manifest IDs intact."""
    train_rows = sorted(data.split_indices["train"])
    fitting = split["train_image_manifest_indices"]
    validation = sorted(split["validation_image_manifest_indices"])
    if (len(train_rows), len(fitting), len(validation)) != (1200, 960, 240):
        raise ValueError("Require the declared 960/240 original-training owner split")
    if len(set(fitting)) != len(fitting) or len(set(validation)) != len(validation):
        raise ValueError("Duplicate inner split owners")
    if set(fitting) & set(validation) or set(fitting) | set(validation) != set(train_rows):
        raise ValueError("Inner split must partition only original training owners")
    source_rows = sorted({j for i in train_rows for j, code in data.pairs[i].items() if code == 1})
    if len(source_rows) != 6000:
        raise ValueError("The complete training gallery must have 6000 source captions")
    row_lookup, text_lookup = {i: k for k, i in enumerate(train_rows)}, {j: k for k, j in enumerate(source_rows)}
    owner = np.full(len(source_rows), -1, dtype=np.int64)
    for image in train_rows:
        for text, code in data.pairs[image].items():
            if code == 1:
                col = text_lookup[text]
                if owner[col] != -1:
                    raise ValueError("A training source caption has multiple owners")
                owner[col] = row_lookup[image]
    if np.any(owner < 0) or not np.array_equal(np.bincount(owner, minlength=1200), np.full(1200, 5)):
        raise ValueError("Every gallery image must own exactly five source captions")
    image_queries = np.asarray([row_lookup[i] for i in validation], dtype=np.int64)
    text_queries = np.flatnonzero(np.isin(owner, image_queries))
    view = SimpleNamespace(**data.__dict__)
    view.split_indices = {"validation": validation}
    return {"data": view, "images": np.asarray(data.images)[train_rows], "texts": np.asarray(data.texts)[source_rows],
            "image_ids": np.asarray(data.image_ids)[train_rows], "text_ids": np.asarray(data.text_ids)[source_rows],
            "owner": owner, "image_queries": image_queries, "text_queries": text_queries,
            "image_manifest_indices": np.asarray(train_rows), "source_text_manifest_indices": np.asarray(source_rows)}


def evaluate_inner(view, scorer=None):
    _, full = exact_retrieval(view["images"], view["texts"], view["owner"], scorer)
    composition, comp = composition_metrics(view["data"], scorer)
    iq, tq = view["image_queries"], view["text_queries"]
    raw = {"gallery_image_ids": view["image_ids"], "gallery_text_ids": view["text_ids"],
           "gallery_owner": view["owner"], "gallery_image_manifest_indices": view["image_manifest_indices"],
           "gallery_source_text_manifest_indices": view["source_text_manifest_indices"],
           "i2t_query_indices": iq, "t2i_query_indices": tq,
           "i2t_cluster_ids": view["image_ids"][iq], "t2i_cluster_ids": view["image_ids"][view["owner"][tq]]}
    for direction, queries in (("i2t", iq), ("t2i", tq)):
        for suffix in ("correct", "top_indices", "top_scores", "candidate_counts", "roundoff_bounds"):
            raw[f"{direction}_{suffix}"] = full[f"{direction}_{suffix}"][queries]
        mask = np.isin(full[f"{direction}_rescored_query_indices"], queries)
        for suffix in ("query_indices", "gallery_indices", "scores"):
            raw[f"{direction}_rescored_{suffix}"] = full[f"{direction}_rescored_{suffix}"][mask]
    raw.update(comp)
    for kind in ("original", "source_pair"):
        raw[f"{kind}_cluster_ids"] = raw[f"{kind}_image_ids"]
        raw[f"{kind}_correct"] = raw[f"{kind}_joint_accuracy"]
    summary = {name: float(raw[f"{name}_correct"].mean()) for name in METRICS}
    summary.update({"gallery_image_count": 1200, "gallery_caption_count": 6000,
                    "image_query_count": len(iq), "caption_query_count": len(tq), "composition": composition,
                    "scope": "fixed_inner_holdout_owners_of_original_training_split_only",
                    "same_score_all_endpoints": True, "original_labels_retained": True})
    return summary, raw


def paired_changes(frozen, trained):
    """Compute paired differences first, avoiding cancellation of scalar means."""
    changes, raw = {}, {}
    for name in METRICS:
        ids = frozen[f"{name}_cluster_ids"]
        baseline = np.asarray(frozen[f"{name}_correct"], dtype=np.float64)
        current = np.asarray(trained[f"{name}_correct"], dtype=np.float64)
        if not np.array_equal(ids, trained[f"{name}_cluster_ids"]) or baseline.shape != current.shape:
            raise ValueError("Paired inner units differ")
        if baseline.ndim != 1 or not len(baseline) or not np.isfinite(baseline).all() or not np.isfinite(current).all():
            raise ValueError("Invalid inner outcome arrays")
        if np.any(baseline < 0) or np.any(baseline > 1) or np.any(current < 0) or np.any(current > 1):
            raise ValueError("Inner accuracies must be in [0,1]")
        if name in ("original", "source_pair"):
            counts = np.asarray(frozen[f"{name}_triplet_counts"], dtype=np.int64)
            if not np.array_equal(counts, trained[f"{name}_triplet_counts"]) or np.any(counts <= 0):
                raise ValueError("Paired composition triplet counts differ")
        else:
            counts = np.ones(len(baseline), dtype=np.int64)
        base_counts, trained_counts = np.rint(baseline * counts).astype(np.int64), np.rint(current * counts).astype(np.int64)
        if not np.array_equal(base_counts / counts, baseline) or not np.array_equal(trained_counts / counts, current):
            raise ValueError("Accuracy is not exactly reconstructible from integer correct/triplet counts")
        numerators = trained_counts - base_counts
        exact = sum((Fraction(int(n), int(d)) for n, d in zip(numerators, counts)), Fraction()) / len(counts)
        difference = numerators.astype(np.float64) / counts
        changes[name] = float(exact)
        raw[f"{name}_difference"] = difference
        raw[f"{name}_cluster_ids"] = np.asarray(ids)
        raw[f"{name}_difference_numerators"] = numerators
        raw[f"{name}_difference_denominators"] = counts
    return changes, raw


def exact_changes(paired):
    result = {}
    for metric in METRICS:
        n, d = paired[f"{metric}_difference_numerators"], paired[f"{metric}_difference_denominators"]
        value = sum((Fraction(int(a), int(b)) for a, b in zip(n, d)), Fraction()) / len(n)
        result[metric] = {"numerator": value.numerator, "denominator": value.denominator}
    return result


def rational(value):
    return Fraction(value["numerator"], value["denominator"])


def inner_eligible(changes, nonzero):
    return bool(nonzero and all(rational(changes[k]) >= 0 for k in METRICS)
                and rational(changes["original"]) > 0)


def select_shared(candidates, *, family, radii=(.1, .3, 1.), weights=(.25, 1., 4.), epochs=(1, 2, 4, 8, 16, 32)):
    """Choose one configuration/epoch shared across encoders, finite coverage."""
    if family not in ("joint", "labclip"):
        raise ValueError("Only declared selectable families are accepted")
    expected = ({(r, w) for r in radii for w in weights} if family == "joint" else {(epoch,) for epoch in epochs})
    indexed = {}
    for row in candidates:
        if row["encoder"] not in ENCODERS:
            raise ValueError("Unexpected encoder")
        key = ((float(row["radius"]), float(row["composition_weight"])) if family == "joint" else (int(row["epoch"]),))
        if key not in expected or (row["encoder"], key) in indexed:
            raise ValueError("Unexpected or duplicate finite-grid state")
        indexed[row["encoder"], key] = row
    if set(indexed) != {(encoder, key) for encoder in ENCODERS for key in expected}:
        raise ValueError("Selection requires complete declared coverage for this family")
    eligible = []
    for key in sorted(expected):
        states = [indexed[encoder, key] for encoder in ENCODERS]
        if not all(inner_eligible(row["exact_paired_changes"], row["nonzero_functional_update"]) for row in states):
            continue
        if family == "labclip" and not all(rational(row["exact_identity_paired_changes"]["original"]) > 0 for row in states):
            continue
        gains = [rational(row["exact_paired_changes"]["original"]) for row in states]
        tail = (-key[0], -abs(math.log(key[1])), -key[1]) if family == "joint" else (-key[0],)
        eligible.append((tuple([min(gains), sum(gains), *tail]), key, states))
    if not eligible:
        return {"passed": False, "eligible_count": 0, "selected_key": None,
                "status": "stop_before_full_fit_and_official_development"}
    _, key, states = max(eligible, key=lambda item: item[0])
    return {"passed": True, "eligible_count": len(eligible), "selected_key": list(key),
            "selected_states": states, "status": "inner_selected_only_not_confirmatory_success"}


def selected_uncertainty(paired):
    effects, samples = {}, {}
    for metric in METRICS:
        effects[metric], samples[metric] = paired_cluster_bootstrap(paired[f"{metric}_difference"], paired[f"{metric}_cluster_ids"])
        effects[metric]["difference"] = float(rational(exact_changes(paired)[metric]))
        effects[metric]["conditioning"] = "one adaptively inner-selected state fixed; inner image clusters resampled; selection uncertainty omitted"
    return effects, samples
