"""Matched v10 causal control: the identical joint learner without retention.

Relative to the frozen v9 fitting loop, this removes the active retention
penalty and every feasibility repair. All objective terms, initial gradient
scaling, optimizer updates, ball radius, seed and epoch selection stay matched.
Final training retention is measured, never represented as a guarantee.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import numpy as np

from .practical_constrained_v8 import _ball


def fit_joint_without_retention(scorer, composition, retrieval, constraints, config, on_epoch=None):
    """Fixed-budget matched control; diagnostics cannot affect fitting/selection."""
    config.validate()
    if scorer.coefficient.shape != (config.rank, config.rank) or np.any(scorer.coefficient != 0):
        raise ValueError("Require an exactly frozen scorer with the configured rank")
    if constraints.gamma != config.retention_fraction:
        raise ValueError("Diagnostic retention fraction differs")
    coefficient = scorer.coefficient.copy()
    initial_r, initial_rg = retrieval.loss_gradient(coefficient)
    initial_c, initial_cg = composition.loss_gradient(coefficient, composition.eligible, config)
    norm_r, norm_c = float(np.linalg.norm(initial_rg)), float(np.linalg.norm(initial_cg))
    if not math.isfinite(norm_r) or norm_r <= 0 or not math.isfinite(norm_c):
        raise ValueError("Invalid or zero initial retrieval gradient")
    multiplier = config.composition_weight * norm_r / max(norm_c, 1e-12)
    if not math.isfinite(multiplier):
        raise ValueError("Invalid fixed composition multiplier")
    initial_objective = initial_r + multiplier * initial_c
    rng, history, best, steps = np.random.default_rng(config.seed), [], None, 0
    baseline = composition.summary(coefficient)
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(constraints.images))
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            _, gradient_r = retrieval.loss_gradient(coefficient, batch)
            _, gradient_c = composition.query_batch_loss_gradient(coefficient, batch, config)
            gradient = gradient_r + multiplier * gradient_c + config.ridge * coefficient
            norm = np.linalg.norm(gradient)
            if not np.isfinite(norm):
                raise FloatingPointError("Nonfinite optimization gradient")
            if norm > config.gradient_clip:
                gradient *= config.gradient_clip / norm
            coefficient -= config.learning_rate / math.sqrt(epoch) * gradient
            coefficient = _ball(coefficient, config.radius)
            steps += 1
        loss_r, gradient_r = retrieval.loss_gradient(coefficient)
        loss_c, gradient_c = composition.loss_gradient(coefficient, composition.eligible, config)
        objective = loss_r + multiplier * loss_c + config.ridge * np.sum(coefficient ** 2) / 2
        full_gradient = gradient_r + multiplier * gradient_c + config.ridge * coefficient
        gap_bound = float(np.sum(full_gradient * coefficient) + config.radius * np.linalg.norm(full_gradient))
        row = {"epoch": epoch, "optimizer_steps": steps, "training_objective": float(objective),
               "retrieval_loss": float(loss_r), "joint_composition_loss": float(loss_c),
               "fixed_composition_multiplier": multiplier, "composition": composition.summary(coefficient),
               "nonzero": bool(np.any(coefficient != 0)), "retention_enforced": False,
               "ball_relaxed_convex_suboptimality_upper_bound": max(0.0, gap_bound)}
        history.append(row)
        if row["nonzero"] and (best is None or objective < best[0]):
            best = (float(objective), epoch, coefficient.copy())
        if on_epoch is not None:
            on_epoch(row, coefficient.copy())
    if best is None:
        raise RuntimeError("No nonzero trained control state")
    scorer.coefficient = best[2]
    diagnostic = constraints.scan(scorer.coefficient, config.feasibility_tolerance, add=False, canonical=True)
    return {"schema": "sanw_joint_without_retention_fit_v10", "config": asdict(config), "selected_epoch": best[1],
            "selection": "minimum_nonzero_training_objective_then_earliest_epoch_without_feasibility_filter",
            "initial_training_objective": initial_objective, "initial_retrieval_loss": initial_r,
            "initial_joint_composition_loss": initial_c, "initial_retrieval_gradient_norm": norm_r,
            "initial_joint_gradient_norm": norm_c, "fixed_composition_multiplier": multiplier,
            "baseline_composition": baseline, "selected_training_objective": best[0],
            "optimizer_steps": steps, "history": history, "final_training_retention_diagnostic": diagnostic,
            "retention_enforced": False, "finite_training_retention_guarantee_claimed": False,
            "exact_objective_convergence_claimed": False, "development_or_test_used": False}
