"""Train-only full-gallery constrained bilinear residuals.

The score is universal: frozen cosine plus a double-centered bilinear residual.
No gallery labels, task names, or dataset-specific switches enter inference.
The finite optimization budget does NOT certify an optimal objective value.
The separate final scanner certifies the stated finite-training constraints.
Nothing here claims those constraints generalize to unseen examples.

All feature inputs must already have the declared canonical normalization.
The implementation uses NumPy float64 and needs no text encoder or GPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
import math

import numpy as np


def _matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    return result


def fixed_pca(values: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Training-only PCA, with deterministic signs and no whitening."""
    values = _matrix(values, "PCA values")
    if rank < 1 or rank > min(values.shape):
        raise ValueError("PCA rank exceeds available rows or dimensions")
    mean = values.mean(axis=0)
    _, _, right = np.linalg.svd(values - mean, full_matrices=False)
    basis = right[:rank].T.copy()
    pivots = np.abs(basis).argmax(axis=0)
    signs = np.sign(basis[pivots, np.arange(rank)])
    basis *= np.where(signs == 0, 1, signs)
    return mean, basis


@dataclass
class ConstrainedBilinearScorer:
    image_mean: np.ndarray
    text_mean: np.ndarray
    image_basis: np.ndarray
    text_basis: np.ndarray
    coefficient: np.ndarray

    def __post_init__(self):
        self.image_mean = np.asarray(self.image_mean, dtype=np.float64)
        self.text_mean = np.asarray(self.text_mean, dtype=np.float64)
        self.image_basis = _matrix(self.image_basis, "image basis")
        self.text_basis = _matrix(self.text_basis, "text basis")
        self.coefficient = _matrix(self.coefficient, "coefficient")
        if self.image_basis.shape[0] != len(self.image_mean) or self.text_basis.shape[0] != len(self.text_mean):
            raise ValueError("Mean and basis dimensions differ")
        if self.coefficient.shape != (self.image_basis.shape[1], self.text_basis.shape[1]):
            raise ValueError("Coefficient and basis dimensions differ")
        if self.image_basis.shape[0] != self.text_basis.shape[0]:
            raise ValueError("Frozen image and text dimensions differ")
        for basis in (self.image_basis, self.text_basis):
            if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10, rtol=1e-10):
                raise ValueError("Basis must have orthonormal columns")
        if not np.isfinite(self.image_mean).all() or not np.isfinite(self.text_mean).all():
            raise ValueError("Means must be finite")

    @classmethod
    def from_training(cls, images, texts, rank=64):
        im, ib = fixed_pca(images, rank)
        tm, tb = fixed_pca(texts, rank)
        return cls(im, tm, ib, tb, np.zeros((rank, rank), dtype=np.float64))

    def image_coordinates(self, images):
        return np.einsum("nd,dr->nr", _matrix(images, "images") - self.image_mean, self.image_basis, optimize=False)

    def text_coordinates(self, texts):
        return np.einsum("nd,dr->nr", _matrix(texts, "texts") - self.text_mean, self.text_basis, optimize=False)

    def score_matrix(self, images, texts):
        images, texts = _matrix(images, "images"), _matrix(texts, "texts")
        return images @ texts.T + self.image_coordinates(images) @ self.coefficient @ self.text_coordinates(texts).T

    def score_pairs(self, images, texts):
        images, texts = _matrix(images, "images"), _matrix(texts, "texts")
        if len(images) != len(texts):
            raise ValueError("Paired row counts differ")
        transformed = np.einsum("nd,dr->nr", self.image_coordinates(images), self.coefficient, optimize=False)
        return np.sum(images * texts, axis=1) + np.sum(transformed * self.text_coordinates(texts), axis=1)

    def unit_input_residual_bound(self):
        """Valid for inputs of norm <=1; accounts for numerical mean norms."""
        return float((1 + np.linalg.norm(self.image_mean)) * (1 + np.linalg.norm(self.text_mean)) * np.linalg.norm(self.coefficient, ord=2))

    def save(self, path: Path):
        np.savez_compressed(path, schema=np.asarray("sanw_constrained_bilinear_v8"), image_mean=self.image_mean,
                            text_mean=self.text_mean, image_basis=self.image_basis,
                            text_basis=self.text_basis, coefficient=self.coefficient)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            if str(z["schema"]) != "sanw_constrained_bilinear_v8":
                raise ValueError("Wrong scorer schema")
            return cls(*(z[k] for k in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient")))


@dataclass(frozen=True)
class FitConfig:
    rank: int = 64
    radius: float = 0.1
    retention_fraction: float = 0.5
    relation_type: str = "supported"
    seed: int = 17
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 0.01
    temperature: float = 0.02
    composition_margin: float = 0.005
    ridge: float = 0.01
    constraint_weight: float = 100.0
    gradient_clip: float = 10.0
    repair_rounds: int = 12
    directional_passes: int = 3
    feasibility_tolerance: float = 1e-12

    def validate(self):
        for key in ("rank", "epochs", "batch_size", "repair_rounds", "directional_passes"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("radius", "learning_rate", "temperature", "ridge", "constraint_weight", "gradient_clip", "feasibility_tolerance"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 < self.retention_fraction <= 1:
            raise ValueError("Retention fraction must be in (0,1]")
        if self.relation_type not in ("supported", "source", "both"):
            raise ValueError("Unknown composition positive relation")
        if self.composition_margin < 0 or not math.isfinite(self.composition_margin):
            raise ValueError("Composition margin must be finite and nonnegative")


@dataclass(frozen=True, order=True)
class Edge:
    direction: int  # 0: image-to-text, 1: text-to-image
    query: int
    negative: int


class FullGalleryConstraints:
    """Protect frozen-correct source queries using every training competitor.

    Input columns are in ascending original manifest order, for tie consistency.
    Tied frozen-correct queries are protected by non-strict margin constraints;
    direct ranking checks additionally verify their deterministic tie outcomes.
    """

    def __init__(self, images, source_texts, owner, scorer, retention_fraction=0.5):
        self.images, self.texts = _matrix(images, "images"), _matrix(source_texts, "source texts")
        self.owner = np.asarray(owner, dtype=np.int64)
        if len(self.owner) != len(self.texts) or np.any(self.owner < 0) or np.any(self.owner >= len(images)):
            raise ValueError("Invalid source ownership")
        if not np.all(np.bincount(self.owner, minlength=len(images)) > 0):
            raise ValueError("Every image needs a source caption")
        if not 0 < retention_fraction <= 1:
            raise ValueError("Invalid retention fraction")
        self.gamma = retention_fraction
        self.scorer = scorer
        self.x, self.y = scorer.image_coordinates(images), scorer.text_coordinates(source_texts)
        self.frozen = self.images @ self.texts.T
        self.owned = np.arange(len(images))[:, None] == self.owner[None, :]
        owned_i, owned_t = np.nonzero(self.owned)
        self.frozen[owned_i, owned_t] = np.sum(self.images[owned_i] * self.texts[owned_t], axis=1)
        self.best_positive = np.where(self.owned, self.frozen, -np.inf).argmax(axis=1)
        from .practical_constrained_evaluation_v8 import exact_retrieval
        _, baseline = exact_retrieval(self.images, self.texts, self.owner)
        self.frozen_i2t = baseline["i2t_top_indices"]
        self.frozen_t2i = baseline["t2i_top_indices"]
        self.protect_i2t = self.owner[self.frozen_i2t] == np.arange(len(images))
        self.protect_t2i = self.frozen_t2i == self.owner
        # Recompute every near-zero baseline margin canonically. This preserves
        # zero feasibility for correct queries whose ownership wins an exact tie.
        near_i = np.abs(self.frozen - self.frozen[np.arange(len(images)), self.best_positive, None]) < 1e-10
        near_t = np.abs(self.frozen - self.frozen[self.owner, np.arange(len(source_texts))][None, :]) < 1e-10
        near_rows, near_cols = np.nonzero(near_i | near_t)
        self.frozen[near_rows, near_cols] = np.sum(self.images[near_rows] * self.texts[near_cols], axis=1)
        self.i_margin = self.frozen[np.arange(len(images)), self.best_positive, None] - self.frozen
        self.t_margin = self.frozen[self.owner, np.arange(len(source_texts))][None, :] - self.frozen
        self.active: set[Edge] = set()

    def residual(self, coefficient):
        return self.x @ coefficient @ self.y.T

    def _slacks(self, coefficient):
        residual = self.residual(coefficient)
        i_delta = residual[np.arange(len(self.images)), self.best_positive, None] - residual
        t_delta = residual[self.owner, np.arange(len(self.texts))][None, :] - residual
        i = (1 - self.gamma) * self.i_margin + i_delta
        t = (1 - self.gamma) * self.t_margin + t_delta
        i[self.owned | ~self.protect_i2t[:, None]] = np.inf
        t[self.owned | ~self.protect_t2i[None, :]] = np.inf
        return residual, i, t

    def scan(self, coefficient, tolerance=1e-12, add=True, canonical=False):
        residual, i, t = self._slacks(coefficient)
        i_worst, t_worst = i.argmin(axis=1), t.argmin(axis=0)
        new_edges = [Edge(0, int(q), int(i_worst[q])) for q in range(len(self.images)) if i[q, i_worst[q]] < -tolerance]
        new_edges += [Edge(1, int(q), int(t_worst[q])) for q in range(len(self.texts)) if t[t_worst[q], q] < -tolerance]
        if add:
            self.active.update(new_edges)
        if canonical:
            from .practical_constrained_evaluation_v8 import CanonicalScorer, exact_retrieval
            model = CanonicalScorer(self.scorer.image_mean, self.scorer.text_mean,
                                    self.scorer.image_basis, self.scorer.text_basis, coefficient)
            _, raw = exact_retrieval(self.images, self.texts, self.owner, model)
            i_correct, t_correct = raw["i2t_correct"], raw["t2i_correct"]
        else:
            scores = self.frozen + residual
            i_correct = self.owner[scores.argmax(axis=1)] == np.arange(len(self.images))
            t_correct = scores.argmax(axis=0) == self.owner
        finite = np.concatenate((i[np.isfinite(i)], t[np.isfinite(t)]))
        minimum = float(finite.min()) if len(finite) else None
        lost_i = int(np.sum(self.protect_i2t & ~i_correct))
        lost_t = int(np.sum(self.protect_t2i & ~t_correct))
        return {"protected_i2t_queries": int(self.protect_i2t.sum()), "protected_t2i_queries": int(self.protect_t2i.sum()),
                "source_gallery_images": len(self.images), "source_gallery_texts": len(self.texts),
                "checked_constraints": int(len(finite)), "violated_constraints": int((finite < -tolerance).sum()),
                "min_constraint_slack": minimum, "active_constraints": len(self.active),
                "lost_frozen_correct_i2t": lost_i, "lost_frozen_correct_t2i": lost_t,
                "current_i2t_correct": int(i_correct.sum()), "current_t2i_correct": int(t_correct.sum()),
                "ranking_checked_canonically": bool(canonical),
                "ranking_preserved": lost_i == 0 and lost_t == 0,
                "feasible_with_tolerance": minimum is None or minimum >= -tolerance}

    def edge_factors(self, edges: Sequence[Edge] | None = None):
        edges = sorted(self.active) if edges is None else list(edges)
        left, right, rhs = [], [], []
        for edge in edges:
            q, n = edge.query, edge.negative
            if edge.direction == 0:
                if not self.protect_i2t[q] or self.owner[n] == q:
                    raise ValueError("Invalid protected image edge")
                left.append(self.x[q]); right.append(self.y[self.best_positive[q]] - self.y[n])
                rhs.append(-(1 - self.gamma) * self.i_margin[q, n])
            elif edge.direction == 1:
                if not self.protect_t2i[q] or self.owner[q] == n:
                    raise ValueError("Invalid protected text edge")
                left.append(self.x[self.owner[q]] - self.x[n]); right.append(self.y[q])
                rhs.append(-(1 - self.gamma) * self.t_margin[n, q])
            else:
                raise ValueError("Unknown edge direction")
        return (np.asarray(left, dtype=np.float64).reshape(-1, self.x.shape[1]),
                np.asarray(right, dtype=np.float64).reshape(-1, self.y.shape[1]), np.asarray(rhs))

    def penalty(self, coefficient):
        """Mean squared positive violation, vectorized without dense normals."""
        left, right, rhs = self.edge_factors()
        if not len(rhs):
            return 0.0, np.zeros_like(coefficient)
        violation = np.maximum(rhs - np.sum((left @ coefficient) * right, axis=1), 0)
        value = np.mean(violation ** 2)
        gradient = -2 * left.T @ (violation[:, None] * right) / len(rhs)
        return float(value), gradient

    def radial_feasibility_factor(self, coefficient):
        """Largest ray contraction satisfying every exact margin inequality.

        This is an algebraic training constraint calculation, never a dev sweep.
        Zero-margin constraints can require factor zero; report that explicitly.
        """
        residual = self.residual(coefficient)
        i_delta = residual[np.arange(len(self.images)), self.best_positive, None] - residual
        t_delta = residual[self.owner, np.arange(len(self.texts))][None, :] - residual
        ratio = 1.0
        for delta, margins, valid in ((i_delta, self.i_margin, ~self.owned & self.protect_i2t[:, None]),
                                      (t_delta, self.t_margin, ~self.owned & self.protect_t2i[None, :])):
            needed = valid & (delta < 0)
            if np.any(needed):
                ratio = min(ratio, float(np.min((1 - self.gamma) * margins[needed] / -delta[needed])))
        return max(0.0, min(1.0, ratio))


def _ball(coefficient, radius):
    norm = np.linalg.norm(coefficient)
    return coefficient * min(1.0, radius / norm) if norm else coefficient.copy()


def repair_feasibility(coefficient, constraints, config):
    """Directional cyclic repair, then exact-ray restoration and direct audit.

    This is a feasibility operation, not a claimed closest-point projection.
    Directional work occurs once per epoch, never inside each minibatch.
    """
    result = _ball(np.asarray(coefficient, dtype=np.float64), config.radius)
    before = constraints.scan(result, config.feasibility_tolerance)
    rounds = 0
    for rounds in range(1, config.repair_rounds + 1):
        scan = constraints.scan(result, config.feasibility_tolerance)
        if scan["feasible_with_tolerance"] and scan["ranking_preserved"]:
            break
        left, right, rhs = constraints.edge_factors()
        for _ in range(config.directional_passes):
            for a, b, target in zip(left, right, rhs):
                violation = target - a @ result @ b
                if violation > 0:
                    denominator = (a @ a) * (b @ b)
                    if denominator > 0:
                        result += (violation / denominator) * np.outer(a, b)
            result = _ball(result, config.radius)
    factor = constraints.radial_feasibility_factor(result)
    # A tiny inward step avoids roundoff at positive boundary constraints.
    if factor < 1 and factor > 0:
        factor = float(np.nextafter(factor, 0.0))
    result *= factor
    certificate = constraints.scan(result, config.feasibility_tolerance, canonical=True)
    # Strict deterministic tie outcomes are stronger than the tolerance check.
    # No candidate is allowed to claim a certificate without satisfying both.
    if not certificate["feasible_with_tolerance"] or not certificate["ranking_preserved"]:
        raise ArithmeticError("Final full-gallery retention certificate failed")
    certificate.update({"coefficient_frobenius_norm": float(np.linalg.norm(result)), "radius": config.radius,
                        "retention_fraction": config.retention_fraction, "numeric_tolerance": config.feasibility_tolerance,
                        "directional_rounds": rounds, "radial_restoration_factor": factor,
                        "violations_before_repair": before["violated_constraints"],
                        "claim_scope": "finite_training_gallery_only_no_transfer_guarantee"})
    return result, certificate


class CompositionExamples:
    """Image-balanced supported/source positives against labelled contradictions."""
    def __init__(self, images, texts, positive_indices, negative_indices, scorer):
        self.images, self.texts = _matrix(images, "images"), _matrix(texts, "texts")
        if len(positive_indices) != len(images) or len(negative_indices) != len(images):
            raise ValueError("Composition index lists must cover each training image")
        self.positive = [np.asarray(x, dtype=np.int64) for x in positive_indices]
        self.negative = [np.asarray(x, dtype=np.int64) for x in negative_indices]
        for indices in self.positive + self.negative:
            if np.any(indices < 0) or np.any(indices >= len(texts)):
                raise ValueError("Invalid composition caption index")
        self.eligible = np.asarray([i for i, (p, n) in enumerate(zip(self.positive, self.negative)) if len(p) and len(n)])
        if not len(self.eligible):
            raise ValueError("No training composition comparisons")
        self.x, self.y = scorer.image_coordinates(images), scorer.text_coordinates(texts)
        self.frozen = [None] * len(images)
        for i in self.eligible:
            self.frozen[i] = (texts[self.positive[i]] @ images[i], texts[self.negative[i]] @ images[i])

    def loss_gradient(self, coefficient, image_indices, config):
        gradient = np.zeros_like(coefficient)
        loss, count = 0.0, 0
        for i in image_indices:
            p, n = self.positive[i], self.negative[i]
            if not len(p) or not len(n):
                continue
            u = self.x[i] @ coefficient
            ps, ns = self.frozen[i]
            positive = ps + self.y[p] @ u
            negative = ns + self.y[n] @ u
            z = (config.composition_margin - positive[:, None] + negative[None, :]) / config.temperature
            loss += float(np.logaddexp(0, z).mean())
            sigmoid = np.exp(-np.logaddexp(0, -z)) / (z.size * config.temperature)
            direction = sigmoid.sum(axis=0) @ self.y[n] - sigmoid.sum(axis=1) @ self.y[p]
            gradient += np.outer(self.x[i], direction)
            count += 1
        if not count:
            return 0.0, gradient
        return loss / count, gradient / count

    def summary(self, coefficient):
        accuracy, margins = [], []
        for i in self.eligible:
            u = self.x[i] @ coefficient
            p, n = self.positive[i], self.negative[i]
            ps, ns = self.frozen[i]
            gap = (ps + self.y[p] @ u)[:, None] - (ns + self.y[n] @ u)[None, :]
            accuracy.append(float((gap > 0).mean())); margins.append(float(gap.mean()))
        return {"eligible_images": len(accuracy), "image_mean_pair_accuracy": float(np.mean(accuracy)),
                "image_mean_pair_margin": float(np.mean(margins)),
                "pair_count": int(sum(len(self.positive[i]) * len(self.negative[i]) for i in self.eligible)),
                "averaging": "pairs_within_image_then_equal_eligible_image_weight"}


def fit(scorer, composition, constraints, config, on_epoch: Callable | None = None):
    """Fixed-budget stochastic projected/penalized descent, train-only selection.

    Each epoch's final feasible state is eligible. Select minimum unpenalized
    training objective, with earliest epoch breaking exact ties. Different seeds
    use genuinely different minibatch permutations; no seed uncertainty claim.
    """
    config.validate()
    if scorer.coefficient.shape != (config.rank, config.rank):
        raise ValueError("Scorer rank differs from configuration")
    if constraints.gamma != config.retention_fraction:
        raise ValueError("Constraint retention fraction differs from configuration")
    if np.any(scorer.coefficient != 0):
        raise ValueError("New fits must begin with the exact frozen scorer")
    rng = np.random.default_rng(config.seed)
    coefficient = scorer.coefficient.copy()
    baseline = composition.summary(coefficient)
    initial_loss, _ = composition.loss_gradient(coefficient, composition.eligible, config)
    history, best, steps = [], None, 0
    constraints.scan(coefficient)
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(composition.eligible)
        for start in range(0, len(order), config.batch_size):
            _, gradient = composition.loss_gradient(coefficient, order[start:start + config.batch_size], config)
            _, penalty_gradient = constraints.penalty(coefficient)
            gradient += config.ridge * coefficient + config.constraint_weight * penalty_gradient
            norm = np.linalg.norm(gradient)
            if norm > config.gradient_clip:
                gradient *= config.gradient_clip / norm
            coefficient -= config.learning_rate / math.sqrt(epoch) * gradient
            coefficient = _ball(coefficient, config.radius)
            steps += 1
        coefficient, certificate = repair_feasibility(coefficient, constraints, config)
        loss, _ = composition.loss_gradient(coefficient, composition.eligible, config)
        objective = loss + config.ridge * np.sum(coefficient ** 2) / 2
        row = {"epoch": epoch, "optimizer_steps": steps, "training_objective": float(objective),
               "training_composition_loss": float(loss), "composition": composition.summary(coefficient),
               "certificate": certificate, "nonzero": bool(np.any(coefficient != 0))}
        history.append(row)
        if row["nonzero"] and (best is None or objective < best[0]):
            best = (float(objective), epoch, coefficient.copy())
        if on_epoch is not None:
            on_epoch(row, coefficient.copy())
    if best is None:
        raise RuntimeError("All trained feasible states are exactly zero; no valid trained candidate")
    scorer.coefficient = best[2]
    return {"schema": "sanw_constrained_fit_v8", "config": asdict(config), "selected_epoch": best[1],
            "selection": "minimum_feasible_nonzero_training_objective_then_earliest_epoch",
            "initial_training_objective": initial_loss, "baseline_composition": baseline,
            "selected_training_objective": best[0], "optimizer_steps": steps, "history": history,
            "final_certificate": constraints.scan(scorer.coefficient, config.feasibility_tolerance, canonical=True),
            "exact_objective_convergence_claimed": False, "development_or_test_used": False}


def radius_headroom(scorer, composition, radii=(0.05, 0.1, 0.2)):
    """Training-only optimistic per-pair bounds, not a fit or success forecast."""
    gaps, direction_norm, image_weights = [], [], []
    for i in composition.eligible:
        p, n = composition.positive[i], composition.negative[i]
        ps, ns = composition.frozen[i]
        gap = ps[:, None] - ns[None, :]
        dy = composition.y[p, None, :] - composition.y[n][None, :, :]
        norms = np.linalg.norm(composition.x[i]) * np.linalg.norm(dy, axis=2)
        gaps.extend(gap.ravel()); direction_norm.extend(norms.ravel())
        image_weights.extend(np.full(gap.size, 1 / (len(composition.eligible) * gap.size)))
    gaps, direction_norm, weights = map(np.asarray, (gaps, direction_norm, image_weights))
    rows = []
    for radius in radii:
        if radius <= 0 or not math.isfinite(radius):
            raise ValueError("Preflight radii must be positive")
        optimistic = gaps + radius * direction_norm
        rows.append({"radius": radius, "optimistic_repairable_failure_mass": float(weights[(gaps <= 0) & (optimistic > 0)].sum()),
                     "unrepairable_failure_mass": float(weights[optimistic <= 0].sum()),
                     "bound_ignores_retention_constraints": True,
                     "max_unit_input_residual_bound": float(radius * (1 + np.linalg.norm(scorer.image_mean)) * (1 + np.linalg.norm(scorer.text_mean)))})
    return {"schema": "sanw_constrained_train_headroom_v8", "scope": "training_only_no_model_fit",
            "baseline": composition.summary(np.zeros_like(scorer.coefficient)), "radii": rows,
            "projected_image_norm_quantiles": np.quantile(np.linalg.norm(composition.x, axis=1), [0, .25, .5, .75, 1]).tolist(),
            "projected_text_norm_quantiles": np.quantile(np.linalg.norm(composition.y, axis=1), [0, .25, .5, .75, 1]).tolist()}
