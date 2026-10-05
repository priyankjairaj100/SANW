"""Expanded-data, explanatory LABCLIP-style fitting; no benchmark selection.

Architecture, HNB updates, Adam, and owner-unique sampling reuse the immutable
v9 implementation. Checkpoint selection uses a fixed training-only batch bank,
not online minibatch losses whose negative draws change from epoch to epoch.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math

import numpy as np
import torch

from .practical_labclip_v9 import (
    LABCLIPConfig, NormalizedFullRankAlignment, canonical_transform,
    epoch_batches, hard_negative_batch_loss, sources_by_owner,
)

AUDIT_SEED = 20261005
WITNESS_SOURCE_COUNT = 128
FUNCTIONAL_TOLERANCE = 1e-12


def fixed_audit_batches(eligible, sources, negatives):
    """All five sources once, fixed unique-owner pools and one negative/source."""
    config = LABCLIPConfig(seed=AUDIT_SEED)
    batches = tuple(tuple(np.asarray(x, dtype=np.int64).copy() for x in batch)
                    for batch in epoch_batches(eligible, sources, negatives, config,
                                               np.random.default_rng(AUDIT_SEED)))
    for batch in batches:
        for value in batch:
            value.setflags(write=False)
    return batches


def audit_bank_record(batches):
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.asarray([len(batch[0])], dtype="<i8").tobytes())
        for value in batch:
            digest.update(np.asarray(value, dtype="<i8").tobytes())
    return {"sha256": digest.hexdigest(), "seed": AUDIT_SEED,
            "batch_count": len(batches),
            "source_rows": sum(len(batch[0]) for batch in batches),
            "serialization": "ordered_batches:little_endian_int64_length_then_image_positive_negative_arrays"}


def fixed_training_objective(model, images, texts, batches):
    """Image-row weighted HNB over the complete fixed eligible-source bank.

    Model forward/CE are float32 as in optimization; scalar accumulation uses
    Python floats in the fixed batch order. This is a finite batch-pool audit
    objective, not an all-negative expectation or full-gallery retrieval loss.
    """
    total, count = 0.0, 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for ii, pp, nn in batches:
            # np.copy avoids torch's warning for read-only immutable indices.
            positive = model(texts[pp.copy()]); negative = model(texts[nn.copy()])
            loss, _ = hard_negative_batch_loss(images[ii.copy()], positive, negative, model.logit_scale)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite fixed LABCLIP audit objective")
            total += len(ii) * float(loss); count += len(ii)
    model.train(was_training)
    if not count:
        raise ValueError("Fixed training objective requires a nonempty audit bank")
    return total / count


def functional_witness(images, texts, source_rows, owner, eligible, weight):
    """A fixed training-only witness of a nonzero score change, not a gate.

    Positive scalar maps leave the normalized direction unchanged. The first
    128 eligible source rows in immutable manifest order are a conservative
    witness set: movement outside these rows alone cannot satisfy this rule.
    """
    source_rows, owner = np.asarray(source_rows), np.asarray(owner)
    positions = np.flatnonzero(np.isin(owner, eligible))[:WITNESS_SOURCE_COUNT]
    if not len(positions):
        raise ValueError("Functional witness requires owned training source pairs")
    rows, owners = source_rows[positions], owner[positions]
    weight = np.asarray(weight, dtype=np.float64)
    transformed = canonical_transform(texts[rows], weight)
    identity = canonical_transform(texts[rows], np.eye(weight.shape[0]))
    change = transformed - identity
    max_direction = float(np.sqrt(np.sum(change * change, axis=1, dtype=np.float64)).max())
    score_changes = np.sum(np.asarray(images[owners], dtype=np.float64) * change, axis=1, dtype=np.float64)
    max_score = float(np.abs(score_changes).max())
    scalar = float(np.trace(weight) / len(weight))
    return {"source_rows": rows.tolist(), "owner_rows": owners.tolist(),
            "pair_count": len(rows), "tolerance": FUNCTIONAL_TOLERANCE,
            "weight_identity_distance": float(np.linalg.norm(weight - np.eye(len(weight)))),
            "non_scalar_weight_norm": float(np.linalg.norm(weight - scalar * np.eye(len(weight)))),
            "max_direction_change_from_identity": max_direction,
            "max_owned_pair_score_change_from_identity": max_score,
            "nonzero_functional_update": bool(max_direction > FUNCTIONAL_TOLERANCE and max_score > FUNCTIONAL_TOLERANCE)}


def select_epoch(rows):
    eligible = [row for row in rows if row["epoch"] > 0 and row["nonzero_functional_update"]
                and math.isfinite(row["training_objective"])]
    if not eligible:
        raise ValueError("No finite functionally nonzero LABCLIP epoch is eligible")
    return min(eligible, key=lambda row: (row["training_objective"], row["epoch"]))["epoch"]


def fit_expanded_alignment(images, texts, source_rows, owner, negatives, eligible,
                           config: LABCLIPConfig, native_logit_scale, on_epoch=None):
    """Fixed 32-epoch inherited optimizer; all epoch states receive diagnostics."""
    config.validate()
    if config.arm != "small_data_fixed_scale":
        raise ValueError("Expanded explanatory arm uses only the inherited fixed-scale recipe")
    torch.set_num_threads(config.threads); torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    images, texts = np.asarray(images, dtype=np.float64), np.asarray(texts, dtype=np.float64)
    if (images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]
            or not np.isfinite(images).all() or not np.isfinite(texts).all()):
        raise ValueError("Require finite image/text matrices with the same feature dimension")
    sources = sources_by_owner(source_rows, owner, len(images))
    eligible = np.asarray(eligible, dtype=np.int64)
    batches = fixed_audit_batches(eligible, sources, negatives)
    bank = audit_bank_record(batches)
    image_tensor = torch.tensor(images, dtype=torch.float32)
    text_tensor = torch.tensor(texts, dtype=torch.float32)
    model = NormalizedFullRankAlignment(images.shape[1], native_logit_scale=native_logit_scale,
                                        learned_scale=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                  betas=(.9, .999), eps=1e-8, weight_decay=0)
    history, checkpoint_rows, steps = [], [], 0

    def assess(epoch, update_row):
        objective = fixed_training_objective(model, image_tensor, text_tensor, batches)
        witness = functional_witness(images, texts, source_rows, owner, eligible,
                                     model.linear.weight.detach().cpu().numpy())
        row = {**update_row, "epoch": epoch, "optimizer_steps": steps,
               "training_objective": objective, "audit_bank_sha256": bank["sha256"],
               "functional_witness": witness,
               "nonzero_functional_update": witness["nonzero_functional_update"],
               "logit_scale": float(model.logit_scale)}
        checkpoint_rows.append(row)
        if on_epoch is not None:
            on_epoch(row, model)

    assess(0, {"online_loss": None, "batches": 0, "caption_rows": 0})
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, count_batches, samples, repeated = 0.0, 0, 0, 0
        for ii, pp, nn in epoch_batches(eligible, sources, negatives, config, rng):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = hard_negative_batch_loss(image_tensor[ii], model(text_tensor[pp]),
                                               model(text_tensor[nn]), model.logit_scale)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite LABCLIP training loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite LABCLIP gradient")
            optimizer.step()
            loss_sum += float(loss.detach()); count_batches += 1; samples += len(ii)
            repeated += len(ii) - len(set(ii.tolist())); steps += 1
        row = {"epoch": epoch, "optimizer_steps": steps, "online_loss": loss_sum / count_batches,
               "batches": count_batches, "caption_rows": samples,
               "repeated_image_rows_within_batches": repeated}
        history.append(row); assess(epoch, row)
    selected = select_epoch(checkpoint_rows)
    return {"config": asdict(config), "audit_bank": bank, "history": history,
            "checkpoint_history": checkpoint_rows, "optimizer_steps": steps,
            "selected_epoch": selected, "selection": "minimum_fixed_training_HNB_objective_then_earliest_epoch_among_nonzero_witness_states",
            "checkpoint_candidates": list(range(1, 33)),
            "heldout_used_in_fitting_or_selection": False}
