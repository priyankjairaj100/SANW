"""Unfitted fallback: train-only, potentially affected source-pair examples.

This module is a training-data/loss replacement, not an inference rule. Every
example contains two distinct source descriptions and one labelled contradiction
from the same training image. No equivalence between the positives is assumed.
The frozen joint margin must lie in the closed interval [-2*epsilon, 2*epsilon].
Eligibility is fixed before fitting; it never depends on the learned scorer.
The closed boundary includes examples that cannot become strictly correct under
the bounded correction; no claim that every included triple is repairable is made.

Use ``SourcePairTrainingExamples(data, hard_count, epsilon)`` in a separately
authorized runner. ``data`` must contain the same already-normalized features
used by the original fitting code. ``batch_loss`` accepts either BoundedPairScorer
or PreparedTextScores through their shared score_pairs/score_matrix interface.
The existing token ``training_caption_indices`` helper remains compatible.

Source ownership and contradiction labels are used only to construct training
losses. Gallery labels never enter the model or its inference score. This module
does not select checkpoints, inspect test data, launch fits, or alter any gate.
Before fitting, record a new protocol covering this objective and its development
selector. The old source/supported development statistic is not silently changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import re
from typing import Mapping, Sequence
import unicodedata

import torch

from .practical_training import (
    PracticalTrainingConfig, TrainingExamples, retained_margin_loss,
    source_retrieval_loss,
)


def caption_key(text: str) -> tuple[str, ...]:
    """Deterministic conflict key: NFKC, casefold, then Unicode word tokens.

    This deliberately ignores punctuation and whitespace. It is not a language
    model tokenizer and does not detect semantic contradiction or annotation noise.
    """
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


@dataclass(frozen=True)
class SourcePairPlan:
    # Global feature-row indices, two distinct positive texts before a negative.
    triplets: tuple[tuple[int, int, int], ...]
    frozen_margins: tuple[float, ...]
    excluded_negative_indices: tuple[int, ...]
    distinct_positive_pair_count: int
    candidate_triplet_count: int

    @property
    def text_indices(self) -> tuple[int, ...]:
        return tuple(sorted({index for triplet in self.triplets for index in triplet}))


def mine_source_pair_plan(
    source_indices: Sequence[int], negative_indices: Sequence[int],
    all_positive_indices: Sequence[int], text_values: Mapping[int, str],
    frozen_scores: Mapping[int, float], epsilon: float,
) -> SourcePairPlan:
    """Enumerate unordered distinct-source pairs and freeze margin eligibility.

    ``all_positive_indices`` includes supported captions solely for conflict
    exclusion. Supported captions never become ranking positives in this loss.
    Negatives matching any declared positive are removed. Positive pairs with
    identical normalized token sequences are omitted, even under different IDs.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("A finite positive frozen score budget is required.")
    sources = sorted(set(source_indices))
    positives = sorted(set(all_positive_indices) | set(sources))
    negatives = sorted(set(negative_indices))
    positive_keys = {caption_key(text_values[index]) for index in positives}
    conflicts = tuple(index for index in negatives
                      if caption_key(text_values[index]) in positive_keys)
    conflict_set = set(conflicts)
    negatives = [index for index in negatives if index not in conflict_set]
    positive_pairs = [(left, right) for left, right in combinations(sources, 2)
                      if caption_key(text_values[left]) != caption_key(text_values[right])]
    required = set(sources) | set(negatives)
    if any(not math.isfinite(float(frozen_scores[index])) for index in required):
        raise ValueError("Mining requires finite frozen scores.")
    triplets, margins = [], []
    for left, right in positive_pairs:
        lower_positive = min(float(frozen_scores[left]), float(frozen_scores[right]))
        for negative in negatives:
            margin = lower_positive - float(frozen_scores[negative])
            if -2 * epsilon <= margin <= 2 * epsilon:
                triplets.append((left, right, negative))
                margins.append(margin)
    return SourcePairPlan(tuple(triplets), tuple(margins), conflicts,
                          len(positive_pairs), len(positive_pairs) * len(negatives))


def source_pair_triplet_loss(
    scores: torch.Tensor, triplet_positions: torch.Tensor, *,
    margin: float, temperature: float,
) -> torch.Tensor:
    """Mean joint ranking loss for an image's explicitly selected triplets."""
    if scores.ndim != 1 or triplet_positions.ndim != 2 or triplet_positions.shape[1] != 3:
        raise ValueError("Require one score vector and an [N,3] triplet index tensor.")
    if len(triplet_positions) == 0:
        return scores.sum() * 0.0
    if temperature <= 0 or not math.isfinite(temperature) or margin < 0 or not math.isfinite(margin):
        raise ValueError("Require finite positive temperature and nonnegative margin.")
    values = scores[triplet_positions]
    first = (values[:, 2] - values[:, 0] + margin) / temperature
    second = (values[:, 2] - values[:, 1] + margin) / temperature
    return torch.logsumexp(torch.stack((torch.zeros_like(first), first, second)), dim=0).mean()


class SourcePairTrainingExamples(TrainingExamples):
    """Drop-in shared loss provider; no scorer or inference modifications."""

    def __init__(self, data, hard_count: int, epsilon: float):
        super().__init__(data, hard_count)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("A finite positive frozen score budget is required.")
        self.epsilon = float(epsilon)
        training_text_indices = {index for image_index in self.image_indices
                                 for index in data.pairs[image_index]}
        text_values = {index: data.manifest["texts"][index]["text"]
                       for index in training_text_indices}
        self.source_pair_plans: list[SourcePairPlan] = []
        # Retain the three-group layout expected by token caption dependency
        # closure. Only eligible source and negative texts are needed here.
        self.composition = []
        for image_index in self.image_indices:
            relations = data.pairs[image_index]
            sources = [index for index, code in relations.items() if code == 1]
            negatives = [index for index, code in relations.items() if code == 3]
            positives = [index for index, code in relations.items() if code in (1, 2)]
            candidate_indices = sorted(set(sources) | set(negatives))
            with torch.no_grad():
                scores = (data.images[image_index].double()[None, :] *
                          data.texts[candidate_indices].double()).sum(dim=-1).tolist()
            plan = mine_source_pair_plan(sources, negatives, positives, text_values,
                                         dict(zip(candidate_indices, scores)), epsilon)
            self.source_pair_plans.append(plan)
            used_sources = sorted({index for triplet in plan.triplets for index in triplet[:2]})
            used_negatives = sorted({triplet[2] for triplet in plan.triplets})
            self.composition.append((used_sources, [], used_negatives))
        self.mining_summary = {
            "scope": "train_only", "epsilon": self.epsilon,
            "closed_frozen_joint_margin_interval": [-2 * self.epsilon, 2 * self.epsilon],
            "training_image_count": len(self.image_indices),
            "eligible_image_count": sum(bool(plan.triplets) for plan in self.source_pair_plans),
            "eligible_triplet_count": sum(len(plan.triplets) for plan in self.source_pair_plans),
            "candidate_triplet_count_after_conflicts": sum(plan.candidate_triplet_count for plan in self.source_pair_plans),
            "excluded_conflicting_negative_count": sum(len(plan.excluded_negative_indices) for plan in self.source_pair_plans),
            "averaging": "mean_eligible_triplets_per_image_then_mean_eligible_images_in_batch",
            "zero_eligible_batch": "zero_composition_loss_retrieval_loss_continues",
            "gallery_labels_at_inference": False,
        }

    def batch_loss(self, model, local_indices: list[int], config: PracticalTrainingConfig):
        if not local_indices:
            raise ValueError("A nonempty training image minibatch is required.")
        if float(config.epsilon) != self.epsilon:
            raise ValueError("The fitting budget differs from the frozen mining budget.")
        image_indices = torch.tensor(local_indices, dtype=torch.long)
        source_indices = torch.tensor(sorted(j for i in local_indices for j in self.source_by_image[i]), dtype=torch.long)
        source_scores = model.score_matrix(self.images[image_indices], self.texts[source_indices], pair_chunk=config.pair_chunk)
        source_mask = image_indices[:, None] == self.owner[source_indices][None, :]
        retrieval = source_retrieval_loss(source_scores, source_mask, self.data.logit_scale)

        comp_images, comp_texts, image_triplets = [], [], []
        for local_index in local_indices:
            plan = self.source_pair_plans[local_index]
            if not plan.triplets:
                continue
            text_indices = plan.text_indices
            lookup = {index: position + len(comp_texts) for position, index in enumerate(text_indices)}
            image_triplets.append(torch.tensor([[lookup[index] for index in triple] for triple in plan.triplets], dtype=torch.long))
            comp_images.extend([self.image_indices[local_index]] * len(text_indices))
            comp_texts.extend(text_indices)
        if image_triplets:
            comp_scores = model.score_pairs(self.data.images[comp_images], self.data.texts[comp_texts])
            composition = torch.stack([
                source_pair_triplet_loss(comp_scores, indices, margin=config.composition_margin,
                                         temperature=config.composition_temperature)
                for indices in image_triplets
            ]).mean()
        else:
            # Keep the same equal-eligible-image averaging convention. There is
            # no division by zero and no invented negative/positive in this case.
            composition = source_scores.sum() * 0.0

        # The following source retention expressions match TrainingExamples.
        # They do not use or depend on composition eligibility.
        i_negative = self.i2t_negative[image_indices]
        i_images = self.images[image_indices].repeat_interleave(i_negative.shape[1], dim=0)
        i_pos_texts = self.texts[self.best_source[image_indices]].repeat_interleave(i_negative.shape[1], dim=0)
        i_pos = model.score_pairs(i_images, i_pos_texts)
        i_neg = model.score_pairs(i_images, self.texts[i_negative.reshape(-1)])
        i_margin = retained_margin_loss(i_pos, i_neg, self.i2t_teacher_margin[image_indices].reshape(-1), cap=config.retention_margin_cap)

        t_negative = self.t2i_negative[source_indices]
        t_texts = self.texts[source_indices].repeat_interleave(t_negative.shape[1], dim=0)
        t_pos_images = self.images[self.owner[source_indices]].repeat_interleave(t_negative.shape[1], dim=0)
        t_pos = model.score_pairs(t_pos_images, t_texts)
        t_neg = model.score_pairs(self.images[t_negative.reshape(-1)], t_texts)
        t_margin = retained_margin_loss(t_pos, t_neg, self.t2i_teacher_margin[source_indices].reshape(-1), cap=config.retention_margin_cap)
        retention_hinge = (i_margin + t_margin) * (self.data.logit_scale / 2)
        total = config.composition_weight * composition + config.retention_weight * (retrieval + retention_hinge)
        values = {"loss": float(total.detach()), "composition_loss": float(composition.detach()),
                  "source_retrieval_ce": float(retrieval.detach()),
                  "retention_hinge_scaled": float(retention_hinge.detach()),
                  "composition_eligible_image_count": len(image_triplets),
                  "composition_eligible_triplet_count": sum(len(indices) for indices in image_triplets)}
        return total, values
