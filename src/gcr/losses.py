"""Reconstructed relation-policy objectives with detached candidate weights.

The historical exact similarity kernel was not recoverable. Its explicitly
specified reconstruction is documented in docs/LOSS_RECONSTRUCTION.md. This
module does not import or tune against any historical result values.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F

UNANNOTATED, SOURCE, SUPPORTED, CONTRADICTED, NEUTRAL = range(5)
METHODS = (
    "clip", "sanw_fixed", "sanw_median", "constant", "shuffled",
    "multipositive", "grounded", "grounded_no_hardening",
    "grounded_no_abstention", "pairwise_rank", "random_exclusion", "smoothing",
)
EXPANDED_METHODS = frozenset({
    "multipositive", "grounded", "grounded_no_hardening",
    "grounded_no_abstention", "random_exclusion",
})


class Policy(NamedTuple):
    positives: Tensor
    weights: Tensor
    smoothing: float


def _check_relations(relations: Tensor) -> None:
    if relations.ndim != 2 or min(relations.shape) == 0:
        raise ValueError("relations must be a nonempty image-by-text matrix")
    if relations.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("relations must have integer dtype")
    if bool(((relations < UNANNOTATED) | (relations > NEUTRAL)).any()):
        raise ValueError("relation codes must be in [0, 4]")


def _randperm(n: int, device: torch.device, generator: torch.Generator | None) -> Tensor:
    # Honor CPU generators even when the score matrix resides on a GPU.
    draw_device = generator.device if generator is not None else device
    return torch.randperm(n, generator=generator, device=draw_device).to(device)


def policy_components(
    relations: Tensor,
    method: str,
    *,
    dtype: torch.dtype = torch.float32,
    semantic_text_features: Tensor | None = None,
    semantic_image_features: Tensor | None = None,
    generator: torch.Generator | None = None,
    semantic_threshold: float = 0.5,
    semantic_softness: float = 0.05,
    negative_weight: float = 0.25,
    hardening_weight: float = 2.0,
    smoothing: float = 0.1,
) -> Policy:
    """Build one image-text policy matrix; transpose it for the reverse loss.

    All source captions are positive. Expanded policies additionally promote
    supported hypotheses. All other relation cells, including missing cross-
    image annotations, start as negative candidates of weight one. Semantic
    controls and smoothing use the source-positive set. Random exclusion uses
    the expanded set and matches each image's number of neutral annotations.
    """
    del semantic_image_features  # Accepted for a shared encoder interface.
    _check_relations(relations)
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    if method == "pairwise_rank":
        raise ValueError("pairwise_rank has no contrastive target policy")
    if not 0 <= smoothing <= 1:
        raise ValueError("smoothing must lie in [0, 1]")
    if negative_weight < 0 or hardening_weight < 0:
        raise ValueError("candidate weights must be nonnegative")
    if semantic_softness <= 0:
        raise ValueError("semantic_softness must be positive")
    with torch.no_grad():
        positives = relations == SOURCE
        if method in EXPANDED_METHODS:
            positives = positives | (relations == SUPPORTED)
        weights = torch.ones(relations.shape, dtype=dtype, device=relations.device)
        negatives = ~positives
        if method in ("grounded", "grounded_no_abstention"):
            weights[relations == CONTRADICTED] = hardening_weight
        if method in ("grounded", "grounded_no_hardening"):
            weights[relations == NEUTRAL] = 0
        if method == "constant":
            weights[negatives] = negative_weight
        if method in ("sanw_fixed", "sanw_median", "shuffled"):
            if semantic_text_features is None:
                raise ValueError(f"{method} requires frozen semantic_text_features")
            if semantic_text_features.ndim != 2 or semantic_text_features.shape[0] != relations.shape[1]:
                raise ValueError("semantic_text_features must have one row per candidate text")
            semantic = F.normalize(semantic_text_features.detach().to(device=relations.device, dtype=dtype), dim=-1)
            if not bool(torch.isfinite(semantic).all()):
                raise ValueError("semantic_text_features must be finite")
            source = relations == SOURCE
            count = source.sum(dim=1, keepdim=True)
            if bool((count == 0).any()):
                raise ValueError("similarity weighting requires a source caption for every image")
            # Mean cosine to the source captions. Do not re-normalize the mean.
            source_mean = source.to(dtype) @ semantic / count.to(dtype)
            similarity = ((source_mean @ semantic.T) + 1) / 2
            threshold = torch.as_tensor(semantic_threshold, dtype=dtype, device=relations.device)
            if method == "sanw_median" and bool(negatives.any()):
                # Statistical median averages the two central values for even N.
                threshold = torch.quantile(similarity[negatives], 0.5)
            weights = torch.sigmoid((threshold - similarity) / semantic_softness)
            if method == "shuffled":
                flat = weights[negatives].clone()
                weights[negatives] = flat[_randperm(flat.numel(), relations.device, generator)]
        if method == "random_exclusion":
            for i in range(relations.shape[0]):
                count = int((relations[i] == NEUTRAL).sum().item())
                candidates = negatives[i].nonzero(as_tuple=False).flatten()
                if count:
                    chosen = candidates[_randperm(candidates.numel(), relations.device, generator)[:count]]
                    weights[i, chosen] = 0
        weights[positives] = 1
    return Policy(positives.detach(), weights.detach(), smoothing if method == "smoothing" else 0.0)


def weighted_row_loss(logits: Tensor, positives: Tensor, weights: Tensor, smoothing: float = 0.0) -> Tensor:
    """Mean uniform-positive cross-entropy over rows with a positive target.

    Zero weights are exact denominator exclusions. Detached weights enter via
    logsumexp(logit + log(weight)); they never introduce a semantic-gradient
    path. Empty-positive rows are skipped. If all rows are ineligible this
    returns a differentiable zero, useful for an empty reverse direction.
    """
    if logits.ndim != 2 or positives.shape != logits.shape or weights.shape != logits.shape:
        raise ValueError("logits, positives, weights must be equally shaped matrices")
    if positives.dtype != torch.bool:
        raise ValueError("positives must be boolean")
    if not 0 <= smoothing <= 1:
        raise ValueError("smoothing must lie in [0, 1]")
    weights = weights.detach().to(device=logits.device, dtype=logits.dtype)
    positives = positives.detach().to(logits.device)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("logits and weights must be finite")
    if bool((weights < 0).any()) or bool((weights[positives] != 1).any()):
        raise ValueError("weights must be nonnegative and one on positives")
    eligible = positives.any(dim=1)
    if not bool(eligible.any()):
        return logits.sum() * 0
    scores, mask, weights = logits[eligible], positives[eligible], weights[eligible]
    log_weights = torch.full_like(weights, -torch.inf)
    nonzero = weights > 0
    log_weights[nonzero] = torch.log(weights[nonzero])
    log_normalizer = torch.logsumexp(scores + log_weights, dim=1)
    target = mask.to(scores.dtype) / mask.sum(dim=1, keepdim=True)
    if smoothing:
        if not bool((weights == 1).all()):
            raise ValueError("candidate-uniform smoothing requires unit candidate weights")
        target = (1 - smoothing) * target + smoothing / scores.shape[1]
    return (log_normalizer - (target * scores).sum(dim=1)).mean()


def symmetric_weighted_loss(logits: Tensor, policy: Policy) -> Tensor:
    """Equal mean of eligible image and text directions, independently normalized."""
    if not bool(policy.positives.any()):
        raise ValueError("contrastive batches must contain at least one positive pair")
    return 0.5 * (
        weighted_row_loss(logits, policy.positives, policy.weights, policy.smoothing)
        + weighted_row_loss(logits.T, policy.positives.T, policy.weights.T, policy.smoothing)
    )


def pairwise_ranking_loss(logits: Tensor, relations: Tensor) -> Tensor:
    """Mean softplus(contradiction - support), with equal weight per eligible image."""
    if logits.shape != relations.shape:
        raise ValueError("logits and relations must have matching shapes")
    row_losses = []
    for scores, labels in zip(logits, relations):
        positive = scores[labels == SUPPORTED]
        negative = scores[labels == CONTRADICTED]
        if positive.numel() and negative.numel():
            row_losses.append(F.softplus(negative[None, :] - positive[:, None]).mean())
    return torch.stack(row_losses).mean() if row_losses else logits.sum() * 0


def contrastive_loss(
    image_features: Tensor,
    text_features: Tensor,
    relations: Tensor,
    method: str,
    logit_scale: float | Tensor,
    generator: torch.Generator | None = None,
    **kwargs,
) -> Tensor:
    """Run a named objective on normalized adapted features and a fixed scale.

    Semantic policies require separately provided frozen semantic_text_features.
    Features are not normalized a second time here. ResidualAdapter does this
    for its outputs. The logit scale, relation masks and weights are detached.
    """
    _check_relations(relations)
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    if image_features.ndim != 2 or text_features.ndim != 2 or image_features.shape[1] != text_features.shape[1]:
        raise ValueError("image and text features must be matrices of equal feature width")
    if (image_features.shape[0], text_features.shape[0]) != relations.shape:
        raise ValueError("relations must match the image and text batch sizes")
    if image_features.device != text_features.device:
        raise ValueError("image and text features must share a device")
    relations = relations.to(image_features.device)
    scale = torch.as_tensor(logit_scale, dtype=image_features.dtype, device=image_features.device).detach()
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("logit_scale must be a finite positive scalar")
    logits = (image_features @ text_features.T) * scale
    if method == "pairwise_rank":
        return pairwise_ranking_loss(logits, relations)
    policy = policy_components(relations, method, dtype=logits.dtype, generator=generator, **kwargs)
    return symmetric_weighted_loss(logits, policy)

