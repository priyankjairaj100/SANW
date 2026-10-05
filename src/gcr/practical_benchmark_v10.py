"""Canonical benchmark outcomes and inherited triplet-weighted clustered inference.

This module contains no dataset loader or model-selection operation. Every pair
uses the same score as official development. SC++ averages over triplet items;
image clusters are retained when resampling, including their unequal sizes.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

from .practical_constrained_evaluation_v8 import exact_retrieval
from .practical_exact_bootstrap_v10 import integer_cluster_bootstrap, lower_exceeds

SEEDS = (17, 29, 43)
ENCODERS = ("vit_b32", "rn50")
DATASETS = ("e_vil_test1000", "coco_karpathy", "sugarcrepe_pp")
REPLICATES, FAMILY_SIZE, BOOTSTRAP_SEED = 100000, 80, 20261007
TAIL = .05 / (2 * FAMILY_SIZE)
CONTRASTS = (("joint_minus_frozen", "joint", "frozen"), ("no_retention_minus_frozen", "no_retention", "frozen"),
             ("joint_minus_no_retention", "joint", "no_retention"))


def score_retrieval(dataset, scorer=None):
    arrays, manifest = dataset["arrays"], dataset["manifest"]
    images, texts = (np.asarray(arrays[key], dtype=np.float64) for key in ("image_features", "text_features"))
    ilook, tlook = ({str(value): i for i, value in enumerate(arrays[key])} for key in ("image_ids", "text_ids"))
    owner = np.full(len(texts), -1, dtype=np.int64)
    for pair in manifest["pairs"]:
        i, j = ilook[str(pair["image_id"])], tlook[str(pair["text_id"])]
        if pair["relation"] != "source" or owner[j] >= 0:
            raise ValueError("Unique source-caption ownership required")
        owner[j] = i
    if np.any(owner < 0):
        raise ValueError("Every gallery caption needs an owner")
    summary, raw = exact_retrieval(images, texts, owner, scorer, query_block=64)
    raw.update(image_ids=arrays["image_ids"], text_ids=arrays["text_ids"], text_source_image_ids=arrays["image_ids"][owner])
    return summary, raw


def score_triplets(dataset, scorer=None):
    arrays, rows = dataset["arrays"], dataset["manifest"]["triplets"]
    ilook, tlook = ({str(value): i for i, value in enumerate(arrays[key])} for key in ("image_ids", "text_ids"))
    ids = [str(row["id"]) for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("SC++ triplet IDs must be nonempty and unique")
    raw = {"item_ids": np.asarray(ids), "image_ids": np.asarray([str(row["image_id"]) for row in rows]),
           "categories": np.asarray([str(row["category"]) for row in rows])}
    images = np.asarray(arrays["image_features"][[ilook[value] for value in raw["image_ids"]]], dtype=np.float64)
    for field in ("positive1", "positive2", "negative"):
        text_ids = np.asarray([str(row[f"{field}_id"]) for row in rows])
        texts = np.asarray(arrays["text_features"][[tlook[value] for value in text_ids]], dtype=np.float64)
        scores = np.sum(images * texts, axis=1, dtype=np.float64) if scorer is None else scorer.pair_scores(images, texts)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite SC++ pair scores")
        raw[f"{field}_ids"], raw[f"{field}_scores"] = text_ids, scores
    raw["positive1_correct"] = raw["positive1_scores"] > raw["negative_scores"]
    raw["positive2_correct"] = raw["positive2_scores"] > raw["negative_scores"]
    raw["correct"] = raw["positive1_correct"] & raw["positive2_correct"]
    return {"items": len(rows), "images": len(set(raw["image_ids"])),
            "positive1_accuracy": float(raw["positive1_correct"].mean()), "positive2_accuracy": float(raw["positive2_correct"].mean()),
            "both_accuracy": float(raw["correct"].mean()), "ties": "fail", "averaging": "equal_triplet_not_equal_image"}, raw


def paired_seed_differences(dataset, metric, selected, reference):
    """Boolean outcomes remain paired by item, gallery, and fixed seed."""
    if set(selected) != set(SEEDS) or set(reference) != set(SEEDS):
        raise ValueError("Require exactly fixed seeds 17,29,43 on both sides")
    caption = dataset == "sugarcrepe_pp"
    field = "correct" if caption else metric.split(".")[0] + "_correct"
    ids_key = "item_ids" if caption else "image_ids" if metric == "i2t.r1" else "text_ids"
    clusters_key = "text_source_image_ids" if metric == "t2i.r1" else "image_ids"
    baseline = reference[17]
    identity_keys = ("item_ids", "image_ids", "categories", "positive1_ids", "positive2_ids", "negative_ids") if caption else (
        "image_ids", "text_ids", "text_source_image_ids", "owner")
    differences = []
    for seed in SEEDS:
        for raw in (selected[seed], reference[seed]):
            if any(not np.array_equal(raw[key], baseline[key]) for key in identity_keys):
                raise ValueError("Paired benchmark identities or gallery changed")
            correct = np.asarray(raw[field])
            if correct.dtype != np.bool_ or correct.shape != baseline[field].shape or correct.ndim != 1 or not len(correct):
                raise ValueError("Benchmark correctness must be a nonempty boolean item vector")
        differences.append(selected[seed][field].astype(np.int64) - reference[seed][field].astype(np.int64))
    delta = np.stack(differences)
    if len(baseline[ids_key]) != delta.shape[1] or len(baseline[clusters_key]) != delta.shape[1]:
        raise ValueError("Cluster/item identities do not cover paired outcomes")
    return delta, baseline[clusters_key]


def bootstrap(delta, clusters):
    """The v6/v7 fixed-seed bootstrap, retaining unequal cluster item counts."""
    delta = np.asarray(delta)
    clusters = np.asarray(clusters)
    if (delta.ndim != 2 or delta.shape != (3, len(clusters)) or not len(clusters)
            or not np.isfinite(delta).all() or not np.isin(delta, (-1, 0, 1)).all()):
        raise ValueError("Require three fixed seeds of paired boolean differences")
    effect, samples = integer_cluster_bootstrap(delta.astype(np.int64).sum(axis=0), 3, clusters)
    effect.update(seed_differences=delta.mean(axis=1).tolist(), images=effect["image_clusters"],
                  conditioning="fixed selected seed mean; image clusters resampled with all owned items; fixed gallery; seeds never resampled")
    return effect, samples


def practical_gate(effects):
    required = {(encoder, dataset, metric) for encoder in ENCODERS for dataset, metric in (
        ("e_vil_test1000", "i2t.r1"), ("e_vil_test1000", "t2i.r1"), ("sugarcrepe_pp", "both_accuracy"),
        ("coco_karpathy", "i2t.r1"), ("coco_karpathy", "t2i.r1"))}
    indexed = {}
    for row in effects:
        if row["contrast"] != "joint_minus_frozen":
            continue
        key = row["encoder"], row["dataset"], row["metric"]
        if key in indexed:
            raise ValueError("Duplicate primary-family effect")
        indexed[key] = row
    if set(indexed) != required:
        raise ValueError("Complete joint-versus-frozen effects required for both encoders")
    gates = {}
    for encoder in ENCODERS:
        retrieval = all(lower_exceeds(indexed[encoder, "e_vil_test1000", metric], Fraction(-1, 100)) for metric in ("i2t.r1", "t2i.r1"))
        caption = lower_exceeds(indexed[encoder, "sugarcrepe_pp", "both_accuracy"], Fraction(0))
        coco = all(lower_exceeds(indexed[encoder, "coco_karpathy", metric], Fraction(-1, 100)) for metric in ("i2t.r1", "t2i.r1"))
        gates[encoder] = {"retrieval_retention": retrieval, "caption_improvement": caption,
                          "passed": retrieval and caption, "secondary_coco_retention": coco}
    return {"encoders": gates, "passed": all(value["passed"] for value in gates.values()),
            "target_family": "joint", "control_families_not_eligible_for_selection": True,
            "historical_test_reuse": "exploratory; family80_does_not_cover_entire_adaptive_history"}
