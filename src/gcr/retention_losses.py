"""Prospectively specified directional objectives and source-gallery distillation.

All image-query columns remain candidates. The directional intervention changes
positive targets and eligible reverse anchors only. Teacher distributions cover
source-caption columns, never hypothesis columns, and are detached.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import SOURCE, SUPPORTED, _check_relations, contrastive_loss, weighted_row_loss


def directional_loss(logits: Tensor, relations: Tensor, *, image_expanded: bool,
                     reverse_expanded: bool) -> Tensor:
    """Equal weighting of the independently normalized image/text directions."""
    _check_relations(relations)
    if logits.shape != relations.shape:
        raise ValueError("logits and relations must have equal shapes")
    source = relations.to(logits.device) == SOURCE
    expanded = source | (relations.to(logits.device) == SUPPORTED)
    if not source.any(dim=1).all():
        raise ValueError("Every image must have at least one source caption")
    weights = torch.ones_like(logits)
    return .5 * (weighted_row_loss(logits, expanded if image_expanded else source, weights)
                 + weighted_row_loss(logits.T, (expanded if reverse_expanded else source).T, weights.T))


def source_distribution_kl(student_logits: Tensor, teacher_logits: Tensor,
                           relations: Tensor, temperature: float = 2.) -> tuple[Tensor, Tensor]:
    """Return KL(teacher||student) in each direction, before T-squared scaling.

    For images the distribution is over every source-caption column in the
    batch. For each source text the distribution is over all batch images.
    An unchanged student gives exactly zero, including in finite precision.
    """
    _check_relations(relations)
    if student_logits.shape != teacher_logits.shape or student_logits.shape != relations.shape:
        raise ValueError("student, teacher and relations must have identical shapes")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("distillation logits must be finite")
    source_columns = (relations.to(student_logits.device) == SOURCE).any(dim=0)
    if not source_columns.any():
        raise ValueError("Source distillation requires source-caption columns")
    student = student_logits[:, source_columns] / temperature
    teacher = teacher_logits.detach().to(student.device)[:, source_columns] / temperature
    def kl(s: Tensor, t: Tensor) -> Tensor:
        log_t = F.log_softmax(t, dim=1)
        return (log_t.exp() * (log_t - F.log_softmax(s, dim=1))).sum(dim=1).mean()
    return kl(student, teacher), kl(student.T, teacher.T)


def retention_loss(images: Tensor, texts: Tensor, relations: Tensor, policy: str,
                   logit_scale: float, *, frozen_images: Tensor | None = None,
                   frozen_texts: Tensor | None = None, temperature: float = 2.) -> Tensor:
    """Frozen protocol loss; native source and supported branches replay exactly."""
    if policy in ("source", "supported"):
        return contrastive_loss(images, texts, relations,
                                "clip" if policy == "source" else "multipositive", logit_scale)
    if policy.startswith("allocation_"):
        source_mix = float(policy.removeprefix("allocation_"))
        if source_mix not in (.5, .8):
            raise ValueError("source mixture outside frozen protocol")
        return (source_mix * retention_loss(images, texts, relations, "source", logit_scale)
                + (1 - source_mix) * retention_loss(images, texts, relations, "supported", logit_scale))
    scale = torch.as_tensor(logit_scale, dtype=images.dtype, device=images.device).detach()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("logit_scale must be a finite positive scalar")
    logits = (images @ texts.T) * scale
    if policy in ("image_source_only", "reverse_source_only"):
        return directional_loss(logits, relations,
                                image_expanded=policy == "reverse_source_only",
                                reverse_expanded=policy == "image_source_only")
    if policy.startswith("distilled_"):
        beta = float(policy.removeprefix("distilled_"))
        if beta not in (1., 4., 16.) or temperature != 2.:
            raise ValueError("distillation coefficient or temperature outside frozen protocol")
        if frozen_images is None or frozen_texts is None:
            raise ValueError("distillation requires detached frozen teacher features")
        # Frozen residual initialization includes this same L2 normalization.
        with torch.no_grad():
            teacher = (F.normalize(frozen_images.detach(), dim=-1)
                       @ F.normalize(frozen_texts.detach(), dim=-1).T) * scale
        image_kl, text_kl = source_distribution_kl(logits, teacher, relations, temperature)
        return (retention_loss(images, texts, relations, "supported", logit_scale)
                + beta * temperature ** 2 * .5 * (image_kl + text_kl))
    raise ValueError(f"Unknown retention policy: {policy}")
