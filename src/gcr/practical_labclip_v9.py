"""Matched-data normalized full-rank LABCLIP controls, on frozen features.

The published-control arm follows the authors' released architecture, HNB loss,
Adam defaults, and caption-row sampling. Its negative data are deliberately our
annotated contradictions, not the paper's synthetic noun/adjective shuffles.
The small-data arm changes only sampling, logit scale, batch size, and budget.
Neither arm promotes supported hypotheses into caption-ownership positives.
All fitting and internal selection use an owner split of ORIGINAL TRAIN ONLY.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



STUDY = "sanw_labclip_inner_v9"
OFFICIAL_FILES = {
    "alignment/learning_alignment.py": "bd0656d3d3d73cc5b77cfc282037a9f2bd56d90f",
    "alignment/coco_alignment.py": "846b0515d34b3e14008f98d84b5ed922a1c9a18e",
    "alignment/alignment_datasets.py": "a610ed0e43f73da5dea8c9fb887854430709d885",
    "alignment/coco_utils.py": "8eacd5c412b5e760375f35abd6a7d0cc455bada1",
}


@dataclass(frozen=True)
class LABCLIPConfig:
    arm: str = "small_data_fixed_scale"
    seed: int = 17
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 32
    threads: int = 3
    source_passes_per_epoch: int = 5
    checkpoint_epochs: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    @classmethod
    def published_control(cls, seed=17, threads=3):
        return cls(arm="published_control", seed=seed, threads=threads,
                   batch_size=1024, epochs=10, checkpoint_epochs=(10,))

    def validate(self):
        if self.arm not in ("published_control", "small_data_fixed_scale"):
            raise ValueError("Unknown LABCLIP control arm")
        for name in ("seed", "batch_size", "epochs", "threads"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate != 0.001:
            raise ValueError("The fixed comparator protocol uses Adam learning_rate=0.001")
        if self.source_passes_per_epoch != 5:
            raise ValueError("The matched recipe visits all five source captions per owner/epoch")
        expected = ((1024, 10, (10,)) if self.arm == "published_control"
                    else (128, 32, (1, 2, 4, 8, 16, 32)))
        if (self.batch_size, self.epochs, tuple(self.checkpoint_epochs)) != expected:
            raise ValueError("Comparator arm differs from its finite, declared recipe")


class NormalizedFullRankAlignment(nn.Module):
    """Pure official scoring: v dot normalize(Wt), with no reference correction."""

    def __init__(self, dimension: int, *, native_logit_scale: float,
                 learned_scale: bool):
        super().__init__()
        if dimension < 1 or not math.isfinite(native_logit_scale) or native_logit_scale <= 0:
            raise ValueError("Require positive dimension and native logit scale")
        self.linear = nn.Linear(dimension, dimension, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dimension, dtype=torch.float32))
        self.learned_scale = learned_scale
        if learned_scale:
            self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer("log_scale", torch.tensor(math.log(native_logit_scale), dtype=torch.float64))

    def forward(self, text):
        transformed = self.linear(text)
        norms = transformed.norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
            raise FloatingPointError("Alignment produced a zero or nonfinite text norm")
        return transformed / norms

    @property
    def logit_scale(self):
        return self.log_scale.exp()

    def canonical_text(self, text):
        """Fixed float64 reductions, independent of inference row batching."""
        return canonical_transform(text, self.linear.weight.detach().cpu().numpy())

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, schema=np.asarray(STUDY),
                            weight=self.linear.weight.detach().cpu().numpy(),
                            log_scale=self.log_scale.detach().cpu().numpy(),
                            learned_scale=np.asarray(self.learned_scale),
                            score=np.asarray("image_dot_l2_normalized_full_rank_transformed_text"))


def canonical_transform(text, weight):
    text, weight = np.asarray(text, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    if text.ndim != 2 or weight.shape != (text.shape[1], text.shape[1]):
        raise ValueError("Expected text matrix and a matching square weight")
    if not np.isfinite(text).all() or not np.isfinite(weight).all():
        raise ValueError("Nonfinite canonical inputs")
    transformed = np.einsum("nd,kd->nk", text, weight, optimize=False)
    norm = np.sqrt(np.sum(transformed * transformed, axis=1, keepdims=True, dtype=np.float64))
    if np.any(norm == 0) or not np.isfinite(norm).all():
        raise FloatingPointError("Canonical alignment produced a zero or nonfinite norm")
    return transformed / norm


def hard_negative_batch_loss(images, positive_text, negative_text, scale):
    """Released HNB objective: Bx2B I2T CE and positive-only BxB T2I CE.

    In particular, negative captions do NOT acquire positive image owners in
    the reverse-direction loss. Other owners' negative captions remain in the
    I2T denominator to retain the comparator's official HNB definition.
    """
    if images.ndim != 2 or positive_text.shape != images.shape or negative_text.shape != images.shape:
        raise ValueError("HNB requires three aligned, nonempty BxD matrices")
    if not len(images):
        raise ValueError("HNB cannot use an empty batch")
    logits = scale * (images @ torch.cat((positive_text, negative_text), dim=0).T)
    labels = torch.arange(len(images), device=images.device)
    i2t = F.cross_entropy(logits, labels)
    t2i = F.cross_entropy(logits[:, :len(images)].T, labels)
    return (i2t + t2i) / 2, {"i2t_ce": i2t.detach(), "t2i_ce": t2i.detach()}


def validate_owner_split(split, training_manifest_indices):
    """Map the shared original-training owner partition into loader positions."""
    if split.get("schema") != "sanw_inner_training_owner_split_v9":
        raise ValueError("Wrong owner split schema")
    train = split.get("train_image_manifest_indices", [])
    validation = split.get("validation_image_manifest_indices", [])
    all_indices = list(training_manifest_indices)
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Training manifest owner indices contain duplicates")
    if len(train) != len(set(train)) or len(validation) != len(set(validation)):
        raise ValueError("The split contains duplicate owners")
    if set(train) & set(validation) or set(train) | set(validation) != set(all_indices):
        raise ValueError("Split must partition original training owners exactly")
    if not train or not validation:
        raise ValueError("Both owner partitions must be nonempty")
    lookup = {value: i for i, value in enumerate(all_indices)}
    return (np.asarray(sorted(lookup[i] for i in train), dtype=np.int64),
            np.asarray(sorted(lookup[i] for i in validation), dtype=np.int64))


def sources_by_owner(source_rows, owner, number_images):
    source_rows, owner = np.asarray(source_rows, dtype=np.int64), np.asarray(owner, dtype=np.int64)
    if source_rows.ndim != 1 or owner.shape != source_rows.shape or len(set(source_rows)) != len(source_rows):
        raise ValueError("Invalid source rows or nonunique caption ownership")
    if np.any(owner < 0) or np.any(owner >= number_images):
        raise ValueError("Source owner is outside the image table")
    sources = [source_rows[owner == i].tolist() for i in range(number_images)]
    if any(not rows for rows in sources):
        raise ValueError("Each image needs at least one source caption")
    return sources


def epoch_batches(train_owner_indices, sources, negatives, config, rng):
    """Yield global loader rows, using no labels from the validation owners.

    The adapted arm makes five passes per epoch for five-source data. Each pass
    has unique-image batches. Source order is permuted once per owner/epoch;
    every source is visited exactly once when source counts are equal.
    """
    train = np.asarray(train_owner_indices, dtype=np.int64)
    if len(set(train)) != len(train) or not len(train):
        raise ValueError("Training owner list must be nonempty and unique")
    if any(not sources[i] or not negatives[i] for i in train):
        raise ValueError("Every fitting owner needs a source and a valid contradiction")
    if config.arm == "published_control":
        rows = [(int(i), int(t)) for i in train for t in sources[i]]
        negative_rng = np.random.default_rng(config.seed + 901)
        fixed_negative = {t: int(negative_rng.choice(negatives[i])) for i, t in rows}
        rows = [rows[j] for j in rng.permutation(len(rows))]
        batches = [rows[start:start + config.batch_size] for start in range(0, len(rows), config.batch_size)]
    else:
        source_orders = {int(i): rng.permutation(sources[i]).tolist() for i in train}
        counts = {len(source_orders[int(i)]) for i in train}
        if counts != {config.source_passes_per_epoch}:
            raise ValueError("The adapted recipe requires exactly five source captions per fitting owner")
        batches = []
        for source_round in range(next(iter(counts))):
            order = rng.permutation(train)
            rows = [(int(i), int(source_orders[int(i)][source_round])) for i in order]
            batches.extend(rows[start:start + config.batch_size] for start in range(0, len(rows), config.batch_size))
    for rows in batches:
        image = np.asarray([i for i, _ in rows], dtype=np.int64)
        positive = np.asarray([t for _, t in rows], dtype=np.int64)
        negative = np.asarray([fixed_negative[int(t)] for t in positive] if config.arm == "published_control"
                              else [rng.choice(negatives[i]) for i in image], dtype=np.int64)
        yield image, positive, negative


def fit_alignment(images, texts, source_rows, owner, negatives, train_owner_indices,
                  config: LABCLIPConfig, native_logit_scale: float,
                  on_checkpoint: Callable | None = None, *, stop_epoch: int | None = None):
    """Fit only; immutable protocol checking and validation are runner duties."""
    config.validate()
    stop_epoch = config.epochs if stop_epoch is None else stop_epoch
    if not isinstance(stop_epoch, int) or stop_epoch < 1 or stop_epoch > config.epochs:
        raise ValueError("Stop epoch must be within the protocol's declared fitting budget")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    images_np, texts_np = np.asarray(images, dtype=np.float64), np.asarray(texts, dtype=np.float64)
    if images_np.ndim != 2 or texts_np.ndim != 2 or images_np.shape[1] != texts_np.shape[1]:
        raise ValueError("Require image and text feature matrices with matching dimensions")
    if not np.isfinite(images_np).all() or not np.isfinite(texts_np).all():
        raise ValueError("Nonfinite fitting inputs")
    sources = sources_by_owner(source_rows, owner, len(images_np))
    # Private copies ensure the historical frozen arrays cannot be mutated.
    image_tensor = torch.tensor(images_np, dtype=torch.float32)
    text_tensor = torch.tensor(texts_np, dtype=torch.float32)
    model = NormalizedFullRankAlignment(images_np.shape[1], native_logit_scale=native_logit_scale,
                                        learned_scale=config.arm == "published_control")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    steps, history = 0, []
    if on_checkpoint is not None:
        on_checkpoint({"epoch": 0, "optimizer_steps": 0, "loss": None, "logit_scale": float(model.logit_scale)}, model)
    for epoch in range(1, stop_epoch + 1):
        model.train()
        loss_sum, batches, samples, repeated = 0.0, 0, 0, 0
        for ii, pp, nn_ in epoch_batches(train_owner_indices, sources, negatives, config, rng):
            optimizer.zero_grad(set_to_none=True)
            positive, negative = model(text_tensor[pp]), model(text_tensor[nn_])
            loss, _ = hard_negative_batch_loss(image_tensor[ii], positive, negative, model.logit_scale)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite comparator training loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite comparator gradient")
            optimizer.step()
            loss_sum += float(loss.detach()); batches += 1; samples += len(ii)
            repeated += len(ii) - len(set(ii.tolist())); steps += 1
        row = {"epoch": epoch, "optimizer_steps": steps, "loss": loss_sum / batches,
               "logit_scale": float(model.logit_scale.detach()), "batches": batches,
               "caption_rows": samples, "repeated_image_rows_within_batches": repeated,
               "weight_identity_distance": float(torch.linalg.vector_norm(model.linear.weight.detach() - torch.eye(images_np.shape[1])))}
        history.append(row)
        if (epoch in config.checkpoint_epochs or epoch == stop_epoch) and on_checkpoint is not None:
            on_checkpoint(row, model)
    return model, {"history": history, "config": asdict(config), "optimizer_steps": steps,
                   "training_owner_count": len(train_owner_indices), "train_source_caption_count": sum(len(sources[i]) for i in train_owner_indices),
                   "supported_captions_used_as_loss_targets": False,
                   "supported_labels_used_for_negative_conflict_exclusion": True, "stop_epoch": stop_epoch}
