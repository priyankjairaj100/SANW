"""Full-gallery retrieval learning plus relation ranking, with v8 certificates.

The scorer and finite-training constraint checker are reused unchanged. This
module changes the objective, capacity, and query sampling. Every sampled query
uses its complete training gallery; gallery negatives are never batch-limited.
No development or benchmark data are loaded here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from .practical_constrained_v8 import FitConfig, _ball, repair_feasibility


@dataclass(frozen=True)
class JointFitConfig(FitConfig):
    rank: int = 128
    radius: float = 0.1
    relation_type: str = "both"
    epochs: int = 32
    learning_rate: float = 0.03
    composition_weight: float = 1.0

    def validate(self):
        super().validate()
        if self.relation_type != "both":
            raise ValueError("V9 requires the actual source-supported joint loss")
        if not math.isfinite(self.composition_weight) or self.composition_weight <= 0:
            raise ValueError("Composition weight must be finite and positive")


def _softmax(logits, axis):
    maximum = np.max(logits, axis=axis, keepdims=True)
    exponential = np.exp(logits - maximum)
    denominator = exponential.sum(axis=axis, keepdims=True)
    return exponential / denominator, np.squeeze(maximum + np.log(denominator), axis=axis)


class FullGallerySourceLoss:
    """Frozen-best-owned I2T anchor versus all unowned captions; owner T2I CE.

    The I2T anchor is fixed before fitting. Other captions owned by the query
    image are excluded from its denominator. The resulting loss is convex in A.
    Every T2I caption has one owning image and compares against every train image.
    Image query batches cover their owned text queries with an unbiased scaling
    even if source-caption counts differ among images.
    """
    def __init__(self, constraints, logit_scale):
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError("Frozen retrieval logit scale must be positive")
        self.c = constraints
        self.scale = float(logit_scale)

    def loss_gradient(self, coefficient, image_indices=None):
        c = self.c
        images = np.arange(len(c.images)) if image_indices is None else np.asarray(image_indices, dtype=np.int64)
        if not len(images) or len(np.unique(images)) != len(images) or np.any(images < 0) or np.any(images >= len(c.images)):
            raise ValueError("Require distinct valid image queries")
        text_queries = np.flatnonzero(np.isin(c.owner, images))
        # Image queries each see the complete source-caption gallery.
        image_logits = self.scale * (c.frozen[images] + (c.x[images] @ coefficient) @ c.y.T)
        anchors = c.best_positive[images]
        other_owned = c.owned[images].copy()
        other_owned[np.arange(len(images)), anchors] = False
        image_logits[other_owned] = -np.inf
        image_prob, image_lse = _softmax(image_logits, 1)
        i_loss = np.mean(image_lse - image_logits[np.arange(len(images)), anchors])
        image_prob[np.arange(len(images)), anchors] -= 1
        image_derivative = self.scale * image_prob / len(images)
        gradient_i = c.x[images].T @ (image_derivative @ c.y)
        # Each owned text query sees the complete training image gallery.
        text_logits = self.scale * (c.frozen[:, text_queries] + (c.x @ coefficient) @ c.y[text_queries].T)
        text_prob, text_lse = _softmax(text_logits, 0)
        owners = c.owner[text_queries]
        denominator = len(images) * len(c.texts) / len(c.images)
        t_loss = np.sum(text_lse - text_logits[owners, np.arange(len(text_queries))]) / denominator
        text_prob[owners, np.arange(len(text_queries))] -= 1
        text_derivative = self.scale * text_prob / denominator
        gradient_t = c.x.T @ (text_derivative @ c.y[text_queries])
        return float((i_loss + t_loss) / 2), (gradient_i + gradient_t) / 2


class JointCompositionExamples:
    """Exactly logsumexp(0, n-source+margin, n-supported+margin) per triple."""
    def __init__(self, images, texts, source_indices, supported_indices, negative_indices, scorer):
        self.images, self.texts = np.asarray(images, dtype=np.float64), np.asarray(texts, dtype=np.float64)
        if any(len(group) != len(images) for group in (source_indices, supported_indices, negative_indices)):
            raise ValueError("Each relation group must cover every image")
        self.source, self.supported, self.negative = [[np.asarray(x, dtype=np.int64) for x in group]
                                                     for group in (source_indices, supported_indices, negative_indices)]
        for indices in self.source + self.supported + self.negative:
            if np.any(indices < 0) or np.any(indices >= len(texts)):
                raise ValueError("Invalid text index")
        self.eligible = np.asarray([i for i in range(len(images))
                                    if len(self.source[i]) and len(self.supported[i]) and len(self.negative[i])])
        if not len(self.eligible):
            raise ValueError("No original joint training triples")
        self.x, self.y = scorer.image_coordinates(images), scorer.text_coordinates(texts)
        self.frozen = [None] * len(images)
        for i in self.eligible:
            self.frozen[i] = tuple(np.sum(texts[ids] * images[i], axis=1)
                                   for ids in (self.source[i], self.supported[i], self.negative[i]))

    def loss_gradient(self, coefficient, image_indices, config):
        loss, count, gradient = 0.0, 0, np.zeros_like(coefficient)
        for i in image_indices:
            if self.frozen[i] is None:
                continue
            source, supported, negative = self.source[i], self.supported[i], self.negative[i]
            a0, b0, n0 = self.frozen[i]
            projected = self.x[i] @ coefficient
            a, b, n = (base + self.y[indices] @ projected
                       for base, indices in ((a0, source), (b0, supported), (n0, negative)))
            first = (n[None, None, :] - a[:, None, None] + config.composition_margin) / config.temperature
            second = (n[None, None, :] - b[None, :, None] + config.composition_margin) / config.temperature
            first, second = np.broadcast_arrays(first, second)
            maximum = np.maximum(0, np.maximum(first, second))
            first_exp, second_exp, zero_exp = np.exp(first - maximum), np.exp(second - maximum), np.exp(-maximum)
            denominator = first_exp + second_exp + zero_exp
            loss += float(np.mean(maximum + np.log(denominator)))
            first_weight = first_exp / denominator / (first.size * config.temperature)
            second_weight = second_exp / denominator / (first.size * config.temperature)
            direction = ((first_weight + second_weight).sum(axis=(0, 1)) @ self.y[negative]
                         - first_weight.sum(axis=(1, 2)) @ self.y[source]
                         - second_weight.sum(axis=(0, 2)) @ self.y[supported])
            gradient += np.outer(self.x[i], direction)
            count += 1
        return (loss / count, gradient / count) if count else (0.0, gradient)

    def summary(self, coefficient):
        accuracy, margins, count = [], [], 0
        for i in self.eligible:
            projected = self.x[i] @ coefficient
            source, supported, negative = self.source[i], self.supported[i], self.negative[i]
            a0, b0, n0 = self.frozen[i]
            a, b, n = (base + self.y[indices] @ projected
                       for base, indices in ((a0, source), (b0, supported), (n0, negative)))
            gap = np.minimum(a[:, None, None], b[None, :, None]) - n[None, None, :]
            accuracy.append(float(np.mean(gap > 0))); margins.append(float(np.mean(gap))); count += gap.size
        return {"eligible_images": len(accuracy), "image_mean_joint_accuracy": float(np.mean(accuracy)),
                "image_mean_joint_margin": float(np.mean(margins)), "triplet_count": count,
                "averaging": "all_source_supported_contradiction_triples_within_image_then_equal_image"}

    def query_batch_loss_gradient(self, coefficient, image_indices, config):
        """Unbiased eligible-image objective when queries sample every owner."""
        if not len(image_indices):
            raise ValueError("Require a nonempty query batch")
        value, gradient = self.loss_gradient(coefficient, image_indices, config)
        eligible_in_batch = sum(self.frozen[i] is not None for i in image_indices)
        scale = len(self.images) * eligible_in_batch / (len(image_indices) * len(self.eligible))
        return value * scale, gradient * scale


def fit_joint(scorer, composition, retrieval, constraints, config, on_epoch=None):
    """A fixed budget of true stochastic query minibatches with full galleries."""
    config.validate()
    if scorer.coefficient.shape != (config.rank, config.rank) or np.any(scorer.coefficient != 0):
        raise ValueError("Require an exactly frozen scorer with the configured rank")
    if constraints.gamma != config.retention_fraction:
        raise ValueError("Retention fraction differs")
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
            _, gradient_p = constraints.penalty(coefficient)
            gradient = gradient_r + multiplier * gradient_c + config.ridge * coefficient + config.constraint_weight * gradient_p
            norm = np.linalg.norm(gradient)
            if not np.isfinite(norm):
                raise FloatingPointError("Nonfinite optimization gradient")
            if norm > config.gradient_clip:
                gradient *= config.gradient_clip / norm
            coefficient -= config.learning_rate / math.sqrt(epoch) * gradient
            coefficient = _ball(coefficient, config.radius)
            steps += 1
        coefficient, certificate = repair_feasibility(coefficient, constraints, config)
        loss_r, gradient_r = retrieval.loss_gradient(coefficient)
        loss_c, gradient_c = composition.loss_gradient(coefficient, composition.eligible, config)
        objective = loss_r + multiplier * loss_c + config.ridge * np.sum(coefficient ** 2) / 2
        full_gradient = gradient_r + multiplier * gradient_c + config.ridge * coefficient
        gap_bound = float(np.sum(full_gradient * coefficient) + config.radius * np.linalg.norm(full_gradient))
        row = {"epoch": epoch, "optimizer_steps": steps, "training_objective": float(objective),
               "retrieval_loss": float(loss_r), "joint_composition_loss": float(loss_c),
               "fixed_composition_multiplier": multiplier, "composition": composition.summary(coefficient),
               "certificate": certificate, "nonzero": bool(np.any(coefficient != 0)),
               "ball_relaxed_convex_suboptimality_upper_bound": max(0.0, gap_bound)}
        history.append(row)
        if row["nonzero"] and (best is None or objective < best[0]):
            best = (float(objective), epoch, coefficient.copy())
        if on_epoch is not None:
            on_epoch(row, coefficient.copy())
    if best is None:
        raise RuntimeError("No nonzero feasible trained state")
    scorer.coefficient = best[2]
    return {"schema": "sanw_joint_fullgallery_fit_v9", "config": asdict(config), "selected_epoch": best[1],
            "selection": "minimum_feasible_nonzero_training_objective_then_earliest_epoch",
            "initial_training_objective": initial_objective, "initial_retrieval_loss": initial_r,
            "initial_joint_composition_loss": initial_c, "initial_retrieval_gradient_norm": norm_r,
            "initial_joint_gradient_norm": norm_c, "fixed_composition_multiplier": multiplier,
            "baseline_composition": baseline, "selected_training_objective": best[0],
            "optimizer_steps": steps, "history": history,
            "final_certificate": constraints.scan(scorer.coefficient, config.feasibility_tolerance, canonical=True),
            "exact_objective_convergence_claimed": False, "development_or_test_used": False}
