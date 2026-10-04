"""Paired inference for the exploratory allocation and distillation extension.

Training seeds are fixed replicates. Their predictions are averaged before image
cluster resampling. Random-assignment draws are equally averaged within a seed.
This module never selects checkpoints from held-out outcomes.
"""
from __future__ import annotations

import numpy as np

from .evaluation import paired_image_bootstrap

SEEDS = (17, 29, 43)
ENCODERS = ("vit_b32", "rn50")
FAMILIES = ("source", "supported", "allocation", "distilled", "allocation_distillation", "wise_ft")
DATASETS = ("e_vil_test1000", "visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
ENDPOINTS = (("e_vil_test1000", "i2t.r1"), ("e_vil_test1000", "t2i.r1"),
             ("visual_entailment", "accuracy"), ("sugarcrepe_pp", "both_accuracy"))
PRIMARY_FAMILY_SIZE = 80
DECOMPOSITION_FAMILY_SIZE = 24
REPLICATES = 100000
BOOTSTRAP_SEED = 20261004
EVIDENCE_TYPE = "exploratory_extension_reusing_previously_examined_test_sets"


def evaluation_specification():
    """The complete evaluation contract, to be frozen before the new grid."""
    return {"status": "frozen", "prior_test_exposure": True, "fresh_confirmatory_claim": False,
        "evidence_type": EVIDENCE_TYPE, "datasets": list(DATASETS), "encoders": list(ENCODERS),
        "primary_endpoints": [{"dataset": dataset, "metric": metric} for dataset, metric in ENDPOINTS],
        "selected_families": list(FAMILIES), "primary_tolerance_pp": 1.,
        "sensitivity_tolerance_pp": 0., "sensitivity_inference": "descriptive_only",
        "primary_contrasts": {"family_size": PRIMARY_FAMILY_SIZE,
            "families_vs_frozen": list(FAMILIES), "joint_vs": ["distilled", "allocation", "wise_ft", "matched_allocation_distillation"],
            "scope": "two encoders times four endpoints; 48 frozen contrasts plus 32 joint contrasts"},
        "decomposition": {"family_size": DECOMPOSITION_FAMILY_SIZE,
            "schedule": "primary AD-selected shared learning rate and per-seed epochs",
            "cells": ["supported", "allocation", "distilled", "allocation_distillation"],
            "parameters": "A uses AD lambda; D uses AD beta; all cells share AD schedule",
            "allocation_main": "0.5*((A-U)+(AD-D))", "distillation_main": "0.5*((D-U)+(AD-A))",
            "interaction": "AD-A-D+U", "conditioning": "conditional on AD development-selected schedule"},
        "inference": {"bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": .05, "interval": "two-sided percentile Bonferroni within each declared family",
            "quantile_method": "linear", "unit": "paired image clusters",
            "training_seed_treatment": "average the three seeds equally before cluster resampling; no seed resampling",
            "random_draw_treatment": "average all three fixed assignment draws equally within each seed",
            "triplet_weighting": "item-weighted mean, recomputed after each image-cluster resample"},
        "practical_success_gate": {"all_three_nonzero_epochs_and_updates": True,
            "i2t_adjusted_lower_bound_strictly_greater_than": -.01,
            "t2i_adjusted_lower_bound_strictly_greater_than": -.01,
            "sugarcrepe_pp_both_adjusted_lower_bound_strictly_greater_than": 0.,
            "confidence_family": PRIMARY_FAMILY_SIZE,
            "cross_encoder_rule": "the same selected family must pass every gate for both encoders"},
        "secondary": "all five datasets and selected 0pp sensitivity strategies reported descriptively",
        "feature_paths": "explicit dataset config hashed in immutable prescore receipt",
        "selection_lock": "both encoders' development selection hashes locked before first held-out score"}


def validate_evaluation_specification(protocol):
    if protocol.get("evaluation") != evaluation_specification():
        raise ValueError("Evaluation specification differs from the frozen implementation contract")


def endpoint_values(prediction, dataset, metric):
    """Return values, item IDs and image cluster IDs without reweighting items."""
    if dataset in ("e_vil_test1000", "coco_karpathy"):
        direction, recall = metric.split(".")
        if direction not in ("i2t", "t2i") or recall not in ("r1", "r5", "r10"):
            raise ValueError("Unsupported retrieval endpoint")
        ids = prediction["image_ids" if direction == "i2t" else "text_ids"]
        clusters = prediction["image_ids" if direction == "i2t" else "text_source_image_ids"]
        values = prediction[f"{direction}_ranks"] <= int(recall[1:])
    elif dataset == "visual_entailment" and metric == "accuracy":
        ids = clusters = prediction["image_ids"]
        values = prediction["image_accuracy"]
    elif dataset in ("sugarcrepe", "sugarcrepe_pp"):
        key = {"accuracy": "correct", "both_accuracy": "correct", "positive1_accuracy": "positive1_correct",
               "positive2_accuracy": "positive2_correct"}[metric]
        ids, clusters, values = prediction["item_ids"], prediction["image_ids"], prediction[key]
    else:
        raise ValueError("Unsupported dataset endpoint")
    ids, clusters, values = np.asarray(ids, str), np.asarray(clusters, str), np.asarray(values, np.float64)
    if ids.ndim != 1 or values.shape != ids.shape or clusters.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError("Malformed endpoint identities")
    if not len(ids) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Malformed endpoint observations")
    return values, ids, clusters


def equal_seed_draw_values(runs, predictions, dataset, metric, *, frozen=False, expected_draws=None):
    """Return [seed,item], with equal fixed-draw means and strict identity checks."""
    if frozen:
        if len(runs) != 1 or runs[0]["method"] != "frozen":
            raise ValueError("A frozen strategy requires exactly one frozen state")
        values, ids, clusters = endpoint_values(predictions[(runs[0]["state_id"], dataset)], dataset, metric)
        return np.repeat(values[None, :], len(SEEDS), axis=0), ids, clusters
    first_ids = first_clusters = None
    output = []
    if {run["seed"] for run in runs} != set(SEEDS):
        raise ValueError("A strategy must contain all three frozen training seeds")
    for seed in SEEDS:
        selected = [run for run in runs if run["seed"] == seed]
        draws = [run.get("draw_id") for run in selected]
        if expected_draws is None:
            if len(selected) != 1 or draws != [None]:
                raise ValueError("A learned strategy requires exactly one state per seed")
        elif sorted(draws) != list(expected_draws):
            raise ValueError("A random strategy requires all fixed draws once per seed")
        values = []
        for run in selected:
            value, ids, clusters = endpoint_values(predictions[(run["state_id"], dataset)], dataset, metric)
            if first_ids is None:
                first_ids, first_clusters = ids, clusters
            elif not np.array_equal(ids, first_ids) or not np.array_equal(clusters, first_clusters):
                raise ValueError("Paired item or image-cluster identities differ")
            values.append(value)
        output.append(np.mean(values, axis=0))
    return np.stack(output), first_ids, first_clusters


def require_aligned(*strategies):
    first = strategies[0]
    for other in strategies[1:]:
        if first[0].shape != other[0].shape or not np.array_equal(first[1], other[1]) or not np.array_equal(first[2], other[2]):
            raise ValueError("Contrast strategies are not paired on identical ordered observations")


def effect(left, right, *, family_size, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    require_aligned(left, right)
    result = paired_image_bootstrap(left[0], right[0], left[2], replicates=replicates,
                                   family_size=family_size, seed=seed, alpha=.05)
    samples = result.pop("bootstrap_differences")
    return result, samples


def decomposition_arrays(u, a, d, ad):
    require_aligned(u, a, d, ad)
    outputs = {"allocation_main": .5 * (a[0] - u[0] + ad[0] - d[0]),
               "distillation_main": .5 * (d[0] - u[0] + ad[0] - a[0]),
               "interaction": ad[0] - a[0] - d[0] + u[0]}
    return {name: (values, u[1], u[2]) for name, values in outputs.items()}


def practical_gate(runs, contrasts):
    """Classify the declared gate. Strict inequalities preserve boundary failures."""
    checks = {"all_three_nonzero": len(runs) == 3 and {r["seed"] for r in runs} == set(SEEDS)
              and all(r["epoch"] > 0 and r["update_norm"] > 0 for r in runs)}
    for name, endpoint, threshold in (("i2t_retention", ("e_vil_test1000", "i2t.r1"), -.01),
                                      ("t2i_retention", ("e_vil_test1000", "t2i.r1"), -.01),
                                      ("sugarcrepe_pp_improvement", ("sugarcrepe_pp", "both_accuracy"), 0.)):
        matches = [row for row in contrasts if (row["dataset"], row["metric"]) == endpoint]
        if len(matches) != 1 or matches[0]["family_size"] != PRIMARY_FAMILY_SIZE:
            raise ValueError("Gate requires each declared adjusted primary contrast once")
        checks[name] = matches[0]["ci_lower"] > threshold
    return {"passed": all(checks.values()), "checks": checks,
            "evidence_type": EVIDENCE_TYPE, "confidence_family_size": PRIMARY_FAMILY_SIZE}
