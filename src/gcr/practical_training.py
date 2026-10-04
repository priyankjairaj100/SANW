"""Development-only fitting of bounded pair-conditioned compatibility scorers.

This new experiment does not modify any earlier scientific implementation.
The shared visual-entailment archive is loaded by the validated existing loader,
but only its train and validation rows are used. No held-out benchmark is read.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .practical_scorer import BoundedPairScorer
from .review_training import SourceRetrievalPool, _atomic_npz
from .training import (
    FeatureDataset, atomic_json, atomic_torch_save, canonical_json,
    seed_everything, sha256_file,
)


@dataclasses.dataclass(frozen=True)
class PracticalTrainingConfig:
    encoder: str = "vit_b32"
    epsilon: float = 0.01
    retention_weight: float = 0.25
    composition_weight: float = 1.0
    seed: int = 17
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    threads: int = 8
    hidden: int = 128
    rank: int = 64
    composition_temperature: float = 0.02
    composition_margin: float = 0.005
    retention_margin_cap: float = 0.01
    hard_negative_count: int = 4
    pair_chunk: int = 8192
    query_chunk: int = 32
    gradient_clip: float = 1.0

    def validate(self) -> None:
        if self.encoder not in ("vit_b32", "rn50"):
            raise ValueError("Only the declared ViT-B/32 and RN50 encoders are supported.")
        for name in ("epsilon", "learning_rate", "composition_temperature", "retention_margin_cap", "gradient_clip"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("retention_weight", "composition_weight", "weight_decay", "composition_margin"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("epochs", "batch_size", "threads", "hidden", "rank", "hard_negative_count", "pair_chunk", "query_chunk"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")


def source_retrieval_loss(scores: torch.Tensor, source_mask: torch.Tensor, logit_scale: float) -> torch.Tensor:
    """Average positive-caption log probabilities, then reverse ownership CE."""
    if scores.ndim != 2 or source_mask.shape != scores.shape:
        raise ValueError("Scores and source mask must have the same matrix shape.")
    if not bool(source_mask.any(dim=1).all()) or not bool((source_mask.sum(dim=0) == 1).all()):
        raise ValueError("Each image needs a source; each source caption needs exactly one owner.")
    logits = scores * logit_scale
    i2t = -((F.log_softmax(logits, dim=1) * source_mask).sum(dim=1) / source_mask.sum(dim=1)).mean()
    t2i = -F.log_softmax(logits, dim=0)[source_mask].mean()
    return (i2t + t2i) / 2


def worst_positive_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, *, margin: float, temperature: float) -> torch.Tensor:
    """Every valid positive is above every contradicted caption at zero loss."""
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        raise ValueError("Composition ranking requires positives and contradictions.")
    return F.softplus((negative_scores - positive_scores.min() + margin) / temperature).mean()


def joint_triplet_loss(source_scores: torch.Tensor, supported_scores: torch.Tensor, negative_scores: torch.Tensor,
                       *, margin: float, temperature: float) -> torch.Tensor:
    """Image-balanced mean loss over source/supported/contradiction triplets.

    logsumexp(0, negative-source, negative-supported) applies pressure to the
    weaker positive within each triplet without selecting one image-global text.
    """
    if min(source_scores.numel(), supported_scores.numel(), negative_scores.numel()) == 0:
        raise ValueError("Joint triplets require source, supported and contradicted captions.")
    source_margin = (negative_scores[None, None, :] - source_scores[:, None, None] + margin) / temperature
    supported_margin = (negative_scores[None, None, :] - supported_scores[None, :, None] + margin) / temperature
    source_margin, supported_margin = torch.broadcast_tensors(source_margin, supported_margin)
    return torch.logsumexp(torch.stack((torch.zeros_like(source_margin), source_margin, supported_margin)), dim=0).mean()


def retained_margin_loss(student_positive: torch.Tensor, student_negative: torch.Tensor, teacher_margin: torch.Tensor, *, cap: float) -> torch.Tensor:
    eligible = teacher_margin > 0
    if not bool(eligible.any()):
        return (student_positive.sum() + student_negative.sum()) * 0.0
    target = teacher_margin[eligible].clamp(max=cap)
    return F.relu(target - (student_positive - student_negative)[eligible]).mean()


def pessimistic_top1(scores: torch.Tensor, indices: torch.Tensor, *, direction: str, owner: torch.Tensor) -> torch.Tensor:
    """Top6 I2T / top2 T2I completely resolve ties for this ownership scheme."""
    if direction == "i2t":
        max_relevant = int(torch.bincount(owner).max())
        if scores.shape[1] < max_relevant + 1:
            raise ValueError("Need more retrieved candidates than possible relevant captions.")
        relevant = owner[indices] == torch.arange(len(scores))[:, None]
    elif direction == "t2i":
        if scores.shape[1] < 2:
            raise ValueError("Need top2 to detect a cross-relevance ownership tie.")
        relevant = indices == owner[:, None]
    else:
        raise ValueError("Unknown retrieval direction.")
    tied = scores == scores[:, :1]
    return relevant[:, 0] & ~(tied & ~relevant).any(dim=1)


def select_development_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Require both development recalls not to drop, then maximize joint gain."""
    baseline = history[0]
    eligible = [row for row in history[1:] if
                row["retrieval"]["image_correct_count"] >= baseline["retrieval"]["image_correct_count"] and
                row["retrieval"]["text_correct_count"] >= baseline["retrieval"]["text_correct_count"] and
                row["optimizer_steps"] > 0]
    if not eligible:
        return {"selected_epoch": None, "eligible_count": 0, "status": "no_nonzero_epoch_passed_development_retention"}
    best = max(eligible, key=lambda row: (
        row["composition"]["paired_joint_accuracy"] - baseline["composition"]["paired_joint_accuracy"],
        row["composition"]["mean_paired_joint_margin"], -row["epoch"]))
    return {"selected_epoch": best["epoch"], "eligible_count": len(eligible),
            "status": "development_selected_not_tested", "checkpoint": best["checkpoint"],
            "paired_joint_accuracy_gain": best["composition"]["paired_joint_accuracy"] - baseline["composition"]["paired_joint_accuracy"],
            "i2t_r1_change": best["retrieval"]["i2t_r1"] - baseline["retrieval"]["i2t_r1"],
            "t2i_r1_change": best["retrieval"]["t2i_r1"] - baseline["retrieval"]["t2i_r1"]}


class TrainingExamples:
    def __init__(self, data: FeatureDataset, hard_count: int):
        self.data = data
        self.image_indices = list(data.split_indices["train"])
        self.images = data.images[self.image_indices]
        self.source_global = sorted({j for i in self.image_indices for j, code in data.pairs[i].items() if code == 1})
        self.texts = data.texts[self.source_global]
        text_lookup = {j: k for k, j in enumerate(self.source_global)}
        self.source_by_image = [[text_lookup[j] for j, code in data.pairs[i].items() if code == 1] for i in self.image_indices]
        owner = torch.full((len(self.texts),), -1, dtype=torch.long)
        for i, entries in enumerate(self.source_by_image):
            for j in entries:
                if owner[j] != -1:
                    raise ValueError("Training source caption has multiple owners.")
                owner[j] = i
        if bool((owner < 0).any()):
            raise ValueError("Unowned source caption.")
        self.owner = owner
        # Hard-negative mining uses training-only frozen source scores.
        frozen = self.images.double() @ self.texts.double().T
        owned = torch.arange(len(self.images))[:, None] == owner[None, :]
        self.best_source = frozen.masked_fill(~owned, -torch.inf).argmax(dim=1)
        self.i2t_negative = frozen.masked_fill(owned, -torch.inf).topk(min(hard_count, len(self.texts) - 5), dim=1).indices
        self.t2i_negative = frozen.masked_fill(owned, -torch.inf).T.topk(min(hard_count, len(self.images) - 1), dim=1).indices
        self.i2t_teacher_margin = frozen[torch.arange(len(self.images)), self.best_source][:, None] - frozen.gather(1, self.i2t_negative)
        self.t2i_teacher_margin = frozen[owner, torch.arange(len(self.texts))][:, None] - frozen.T.gather(1, self.t2i_negative)
        self.composition = []
        for image_index in self.image_indices:
            sources = [j for j, code in data.pairs[image_index].items() if code == 1]
            supported = [j for j, code in data.pairs[image_index].items() if code == 2]
            negatives = [j for j, code in data.pairs[image_index].items() if code == 3]
            self.composition.append((sources, supported, negatives))

    def batch_loss(self, model: BoundedPairScorer, local_indices: list[int], config: PracticalTrainingConfig) -> tuple[torch.Tensor, dict[str, float]]:
        image_indices = torch.tensor(local_indices, dtype=torch.long)
        source_indices = torch.tensor(sorted(j for i in local_indices for j in self.source_by_image[i]), dtype=torch.long)
        source_scores = model.score_matrix(self.images[image_indices], self.texts[source_indices], pair_chunk=config.pair_chunk)
        source_mask = image_indices[:, None] == self.owner[source_indices][None, :]
        retrieval = source_retrieval_loss(source_scores, source_mask, self.data.logit_scale)

        comp_images, comp_texts, spans = [], [], []
        offset = 0
        for local_index in local_indices:
            sources, supported, negatives = self.composition[local_index]
            if not sources or not supported or not negatives:
                continue
            indices = sources + supported + negatives
            comp_images.extend([self.image_indices[local_index]] * len(indices))
            comp_texts.extend(indices)
            spans.append((offset, offset + len(sources), offset + len(sources) + len(supported), offset + len(indices)))
            offset += len(indices)
        if not spans:
            raise ValueError("Batch has no valid positive-versus-contradiction comparisons.")
        comp_scores = model.score_pairs(self.data.images[comp_images], self.data.texts[comp_texts])
        composition = torch.stack([joint_triplet_loss(comp_scores[a:b], comp_scores[b:c], comp_scores[c:d], margin=config.composition_margin,
                                                     temperature=config.composition_temperature) for a, b, c, d in spans]).mean()

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
                  "source_retrieval_ce": float(retrieval.detach()), "retention_hinge_scaled": float(retention_hinge.detach())}
        return total, values


@torch.inference_mode()
def evaluate_development(model: BoundedPairScorer, data: FeatureDataset, pool: SourceRetrievalPool,
                         output: Path, epoch: int, config: PracticalTrainingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    indices = data.split_indices["validation"]
    image_ids, worst_positive, best_negative, joint, source_only, supported_only = [], [], [], [], [], []
    paired_accuracy, paired_margin, triplet_counts = [], [], []
    raw_image, raw_text, raw_relation, raw_score = [], [], [], []
    for i in indices:
        text_indices = sorted(data.pairs[i])
        labels = torch.tensor([data.pairs[i][j] for j in text_indices])
        scores = model.score_pairs(data.images[i:i+1].expand(len(text_indices), -1), data.texts[text_indices])
        raw_image.extend([i] * len(text_indices)); raw_text.extend(text_indices)
        raw_relation.extend(labels.tolist()); raw_score.extend(scores.tolist())
        positive, negative = scores[(labels == 1) | (labels == 2)], scores[labels == 3]
        if not len(positive) or not len(negative):
            continue
        source = scores[labels == 1]
        supported = scores[labels == 2]
        if not len(source) or not len(supported):
            continue
        triplet_margins = torch.minimum(source[:, None, None], supported[None, :, None]) - negative[None, None, :]
        paired_accuracy.append(float((triplet_margins > 0).double().mean()))
        paired_margin.append(float(triplet_margins.mean()))
        triplet_counts.append(triplet_margins.numel())
        low, high = positive.min(), negative.max()
        image_ids.append(data.image_ids[i]); worst_positive.append(float(low)); best_negative.append(float(high))
        joint.append(bool(low > high)); source_only.append(bool(source.min() > high))
        supported_only.append(bool(supported.min() > high) if len(supported) else False)
    if not image_ids:
        raise ValueError("No development composition comparisons.")
    comp_summary = {"image_count": len(image_ids), "all_pairs_correct_count": sum(joint),
                    "all_pairs_accuracy": float(np.mean(joint)), "source_all_correct_count": sum(source_only),
                    "supported_all_correct_count": sum(supported_only),
                    "paired_joint_accuracy": float(np.mean(paired_accuracy)),
                    "mean_paired_joint_margin": float(np.mean(paired_margin)),
                    "triplet_count": sum(triplet_counts), "averaging": "equal_image_weight_then_all_source_supported_contradiction_triplets",
                    "mean_worst_positive_margin": float(np.mean(np.asarray(worst_positive) - np.asarray(best_negative))),
                    "tie_rule": "strict_positive_over_every_contradiction", "neutral_as_negative": False}
    comp_path = output / "development" / f"epoch_{epoch:02d}_composition.npz"
    _atomic_npz(comp_path, image_ids=np.asarray(image_ids), worst_positive_score=np.asarray(worst_positive),
                best_negative_score=np.asarray(best_negative), joint_correct=np.asarray(joint, dtype=bool),
                paired_joint_accuracy=np.asarray(paired_accuracy), paired_joint_margin=np.asarray(paired_margin),
                triplet_count=np.asarray(triplet_counts),
                raw_image_index=np.asarray(raw_image), raw_text_index=np.asarray(raw_text),
                raw_relation=np.asarray(raw_relation), raw_score=np.asarray(raw_score))

    image_scores, image_top = model.exact_topk(pool.images, pool.texts, k=1, direction="i2t",
                                            query_chunk=config.query_chunk, pair_chunk=config.pair_chunk)
    text_scores, text_top = model.exact_topk(pool.images, pool.texts, k=1, direction="t2i",
                                          query_chunk=config.query_chunk, pair_chunk=config.pair_chunk)
    image_correct = pool.owner[image_top[:, 0]] == torch.arange(len(pool.images))
    text_correct = text_top[:, 0] == pool.owner
    ret_summary = {"i2t_r1": float(image_correct.double().mean()), "t2i_r1": float(text_correct.double().mean()),
                   "image_correct_count": int(image_correct.sum()), "text_correct_count": int(text_correct.sum()),
                   "image_count": len(pool.images), "text_count": len(pool.texts),
                   "tie_rule": "descending_float64_score_then_ascending_gallery_manifest_index", "relevance": "source_caption_ownership"}
    ret_path = output / "development" / f"epoch_{epoch:02d}_retrieval.npz"
    _atomic_npz(ret_path, image_ids=np.asarray(pool.image_ids), text_ids=np.asarray(pool.text_ids),
                owner=pool.owner.numpy(), image_correct=image_correct.numpy(), text_correct=text_correct.numpy(),
                image_top_scores=image_scores.numpy(), image_top_indices=image_top.numpy(),
                text_top_scores=text_scores.numpy(), text_top_indices=text_top.numpy())
    comp_summary["predictions"] = {"path": str(comp_path.relative_to(output)), "sha256": sha256_file(comp_path)}
    ret_summary["predictions"] = {"path": str(ret_path.relative_to(output)), "sha256": sha256_file(ret_path)}
    return comp_summary, ret_summary


def input_paths(repository: Path, encoder: str) -> dict[str, Path]:
    if encoder == "vit_b32":
        features = repository / "results/resume_features/visual_entailment"
        development = repository / "results/review_followup/features/e_vil_dev900"
    elif encoder == "rn50":
        features = repository / "results/strengthen_second_encoder/features/visual_entailment"
        development = repository / "results/strengthen_second_encoder/features/review_followup/e_vil_dev900"
    else:
        raise ValueError("Unknown encoder.")
    return {"manifest": repository / "data/visual_entailment/manifest.json", "features": features / "features.npz",
            "metadata": features / "metadata.json", "development_manifest": repository / "data/review_followup/e_vil_dev900/manifest.json",
            "development_features": development / "features.npz", "development_metadata": development / "metadata.json"}


def run_training(repository: Path, output: Path, config: PracticalTrainingConfig) -> dict[str, Any]:
    config.validate()
    repository, output = repository.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Run output already contains files; choose a new run directory.")
    torch.set_num_threads(config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed_everything(config.seed, config.threads)
    paths = input_paths(repository, config.encoder)
    dimension = 512 if config.encoder == "vit_b32" else 1024
    data = FeatureDataset(repository, paths["manifest"], paths["features"], paths["metadata"], dimension)
    pool = SourceRetrievalPool(repository, paths["development_manifest"], paths["development_features"], paths["development_metadata"], data, dimension)
    # Match the frozen ResidualAdapter forward normalization exactly once.
    # Shared-cache test/calibration rows are not used for means, losses or metrics.
    data.images = F.normalize(data.images, dim=-1)
    data.texts = F.normalize(data.texts, dim=-1)
    pool.images = F.normalize(pool.images, dim=-1)
    pool.texts = F.normalize(pool.texts, dim=-1)
    train = TrainingExamples(data, config.hard_negative_count)
    train_text_indices = sorted({j for i in data.split_indices["train"] for j in data.pairs[i]})
    model = BoundedPairScorer(dimension=dimension, epsilon=config.epsilon, hidden=config.hidden, rank=config.rank,
                             image_mean=train.images.mean(dim=0), text_mean=data.texts[train_text_indices].mean(dim=0))
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    source_names = ("src/gcr/practical_training.py", "src/gcr/practical_scorer.py", "scripts/run_practical_training.py",
                    "src/gcr/training.py", "src/gcr/review_training.py")
    identity = {"schema": "sanw_practical_development_v1", "config": dataclasses.asdict(config),
                "inputs": {name: {"path": str(path.relative_to(repository)), "sha256": sha256_file(path)} for name, path in paths.items()},
                "source_sha256": {name: sha256_file(repository / name) for name in source_names},
                "model": model.config(), "training_images": len(train.images), "relation_development_images": len(data.split_indices["validation"]),
                "retrieval_development_images": len(pool.images), "selection_rule": "both_R1_counts_at_least_frozen_then_max_image_averaged_joint_triplet_accuracy_gain_then_mean_triplet_margin_then_earliest_epoch",
                "objective": "composition_weight * image_mean_triplet_logsumexp_0_negative_minus_source_negative_minus_supported + retention_weight * (source_only_bidirectional_CE + frozen_logit_scale * mean_bidirectional_margin_hinge)",
                "feature_preprocessing": "float32 F.normalize once, followed by training-only centering within residual network",
                "held_out_evaluation": "prohibited_by_runner", "neutral_as_negative": False,
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    protocol = repository / "results/practical_v6/protocol_v1.json"
    if protocol.exists():
        identity["protocol"] = {"path": str(protocol.relative_to(repository)), "sha256": sha256_file(protocol)}
    digest = hashlib.sha256(canonical_json(identity)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": digest})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                 foreach=False, fused=False)
    history, steps = [], 0
    permutation_generator = torch.Generator().manual_seed(config.seed)
    started = time.monotonic()
    for epoch in range(config.epochs + 1):
        epoch_start = time.monotonic()
        train_summary = {}
        if epoch:
            model.train()
            permutation = torch.randperm(len(train.images), generator=permutation_generator).tolist()
            totals, total_images = {}, 0
            for offset in range(0, len(permutation), config.batch_size):
                batch = permutation[offset:offset + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss, values = train.batch_loss(model, batch, config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                total_images += len(batch)
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value * len(batch)
            train_summary = {name: value / total_images for name, value in totals.items()}
        checkpoint = output / "checkpoints" / f"epoch_{epoch:02d}.pt"
        update_norm = float(torch.sqrt(sum((parameter.detach() - initial_parameters[name]).double().square().sum()
                                          for name, parameter in model.named_parameters())))
        atomic_torch_save(checkpoint, {"schema": "sanw_practical_pair_scorer_v1", "model_config": model.config(),
                                      "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                                      "seed": config.seed, "update_norm": update_norm,
                                      "training_metadata": {"path": str(output / "ledger.json"), "sha256": sha256_file(output / "ledger.json")},
                                      "config": dataclasses.asdict(config), "epoch": epoch, "optimizer_steps": steps,
                                      "ledger_sha256": digest, "torch_rng_state": torch.get_rng_state(),
                                      "permutation_rng_state": permutation_generator.get_state()})
        composition, retrieval = evaluate_development(model, data, pool, output, epoch, config)
        row = {"epoch": epoch, "optimizer_steps": steps, "update_norm": update_norm, "training": train_summary,
               "composition": composition, "retrieval": retrieval,
               "checkpoint": {"path": str(checkpoint.relative_to(output)), "sha256": sha256_file(checkpoint)},
               "epoch_seconds": time.monotonic() - epoch_start, "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        atomic_json(output / "history.json", {"ledger_sha256": digest, "epochs": history})
        atomic_json(output / "selection.json", select_development_epoch(history))
        print(json.dumps({"event": "epoch_complete", "encoder": config.encoder, "epsilon": config.epsilon,
                          "retention_weight": config.retention_weight, "epoch": epoch, "steps": steps,
                          "paired_joint_accuracy": composition["paired_joint_accuracy"], "i2t_r1": retrieval["i2t_r1"],
                          "t2i_r1": retrieval["t2i_r1"], "epoch_seconds": row["epoch_seconds"]}), flush=True)
    completion = {"status": "completed_development_only", "ledger_sha256": digest,
                  "epochs": config.epochs, "optimizer_steps": steps, "selection": select_development_epoch(history),
                  "history_sha256": sha256_file(output / "history.json"), "elapsed_seconds": time.monotonic() - started}
    atomic_json(output / "completion.json", completion)
    return completion
