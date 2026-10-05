"""Explanatory retrieval-only control; never a selectable practical candidate.

The v10 data, geometry, full galleries, finite-training retention, stochastic
query order and fixed budget are inherited. Only the joint semantic objective
coefficient changes to exactly zero. Composition summaries are descriptive.
"""
from dataclasses import asdict
import math
import numpy as np

from .practical_constrained_v8 import _ball, repair_feasibility


def fit_retrieval_only(scorer, composition, retrieval, constraints, config, on_epoch=None):
    config.validate()
    if scorer.coefficient.shape != (config.rank, config.rank) or np.any(scorer.coefficient != 0):
        raise ValueError("Require an exactly frozen scorer with the inherited rank")
    if constraints.gamma != config.retention_fraction:
        raise ValueError("Retention fraction differs")
    coefficient = scorer.coefficient.copy()
    initial_r, initial_rg = retrieval.loss_gradient(coefficient)
    initial_c, initial_cg = composition.loss_gradient(coefficient, composition.eligible, config)
    norm_r, norm_c = float(np.linalg.norm(initial_rg)), float(np.linalg.norm(initial_cg))
    if not math.isfinite(norm_r) or norm_r <= 0 or not math.isfinite(norm_c):
        raise ValueError("Invalid initial training gradients")
    rng, history, best, steps = np.random.default_rng(config.seed), [], None, 0
    baseline = composition.summary(coefficient)
    for epoch in range(1, config.epochs+1):
        order = rng.permutation(len(constraints.images))
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start+config.batch_size]
            _, gradient_r = retrieval.loss_gradient(coefficient, batch)
            _, gradient_p = constraints.penalty(coefficient)
            gradient = gradient_r + config.ridge*coefficient + config.constraint_weight*gradient_p
            norm = np.linalg.norm(gradient)
            if not np.isfinite(norm):
                raise FloatingPointError("Nonfinite optimization gradient")
            if norm > config.gradient_clip:
                gradient *= config.gradient_clip/norm
            coefficient -= config.learning_rate/math.sqrt(epoch)*gradient
            coefficient = _ball(coefficient, config.radius)
            steps += 1
        coefficient, certificate = repair_feasibility(coefficient, constraints, config)
        loss_r, gradient_r = retrieval.loss_gradient(coefficient)
        loss_c, _ = composition.loss_gradient(coefficient, composition.eligible, config)
        objective = loss_r + config.ridge*np.sum(coefficient**2)/2
        full_gradient = gradient_r + config.ridge*coefficient
        gap_bound = float(np.sum(full_gradient*coefficient)+config.radius*np.linalg.norm(full_gradient))
        row = {"epoch": epoch, "optimizer_steps": steps, "training_objective": float(objective),
               "retrieval_loss": float(loss_r), "joint_composition_loss": float(loss_c),
               "fixed_composition_multiplier": 0.0, "composition": composition.summary(coefficient),
               "certificate": certificate, "nonzero": bool(np.any(coefficient != 0)),
               "ball_relaxed_convex_suboptimality_upper_bound": max(0., gap_bound)}
        history.append(row)
        if row["nonzero"] and (best is None or objective < best[0]):
            best = (float(objective), epoch, coefficient.copy())
        if on_epoch is not None:
            on_epoch(row, coefficient.copy())
    if best is None:
        raise RuntimeError("No nonzero feasible control state")
    scorer.coefficient = best[2]
    return {"schema": "sanw_retrieval_only_fit_v10", "config": asdict(config), "selected_epoch": best[1],
            "selection": "minimum_feasible_nonzero_training_objective_then_earliest_epoch",
            "initial_training_objective": float(initial_r), "initial_retrieval_loss": float(initial_r),
            "initial_joint_composition_loss": float(initial_c), "initial_retrieval_gradient_norm": norm_r,
            "initial_joint_gradient_norm": norm_c, "fixed_composition_multiplier": 0.0,
            "baseline_composition": baseline, "selected_training_objective": best[0],
            "optimizer_steps": steps, "history": history,
            "final_certificate": constraints.scan(scorer.coefficient, config.feasibility_tolerance, canonical=True),
            "explanatory_control_only": True, "candidate_selection_allowed": False,
            "exact_objective_convergence_claimed": False, "development_or_test_used": False}
