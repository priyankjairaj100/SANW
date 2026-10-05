"""Deterministic, validation-only adapter fitting with an immutable run ledger.

Held-out benchmark evaluation deliberately lives in a different module and CLI.
Historical tables are never read by this code.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import torch

from .adapters import ResidualAdapter
from .losses import contrastive_loss


METHODS = (
    "clip", "sanw_fixed", "sanw_median", "constant", "shuffled",
    "multipositive", "grounded", "grounded_no_hardening",
    "grounded_no_abstention", "pairwise_rank", "random_exclusion", "smoothing",
)
RELATIONS = {"source": 1, "supported": 2, "contradicted": 3, "neutral": 4}


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    methods: tuple[str, ...] = METHODS
    learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 10
    image_batch_size: int = 32
    feature_dim: int = 512
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    adam_amsgrad: bool = False
    adam_foreach: bool = False
    adam_fused: bool = False
    grad_clip_norm: float = 1.0
    threads: int = 2
    optimizer: str = "AdamW"
    semantic_threshold: float = 0.5
    semantic_softness: float = 0.05
    negative_weight: float = 0.25
    hardening_weight: float = 2.0
    smoothing: float = 0.1
    validation_positives: tuple[str, ...] = ("source", "supported")
    validation_tie_rule: str = "lowest_manifest_text_index"
    relation_tie_credit: float = 0.5
    selection_tie_rule: str = "earlier_epoch_then_lower_learning_rate"
    device: str = "cpu"

    def validate(self) -> None:
        if self.methods != METHODS:
            raise ValueError("The study ledger must contain all twelve canonical methods.")
        if len(set(self.learning_rates)) != len(self.learning_rates) or any(x <= 0 for x in self.learning_rates):
            raise ValueError("Learning rates must be distinct and positive.")
        if tuple(sorted(self.learning_rates)) != self.learning_rates:
            raise ValueError("Learning rates must be ascending for the declared tie rule.")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("Seeds must be distinct and nonempty.")
        if self.epochs < 0 or self.image_batch_size <= 0 or self.threads <= 0:
            raise ValueError("Invalid epochs, batch size, or thread count.")
        if self.device != "cpu":
            raise ValueError("This reconstruction declares deterministic CPU fitting.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".tmp-", suffix=".json", delete=False) as stream:
        name = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(value, name)
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class FeatureDataset:
    """A validated feature cache and sparse image/text relation annotations."""

    def __init__(self, repository: Path, manifest_path: Path, features_path: Path, metadata_path: Path, dimension: int = 512):
        self.repository = repository.resolve()
        self.manifest_path = manifest_path.resolve()
        self.features_path = features_path.resolve()
        self.metadata_path = metadata_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.metadata = json.loads(self.metadata_path.read_text())
        with np.load(self.features_path, allow_pickle=False) as data:
            self.images = torch.from_numpy(np.array(data["image_features"], dtype=np.float32, copy=True))
            self.texts = torch.from_numpy(np.array(data["text_features"], dtype=np.float32, copy=True))
            self.image_ids = [str(x) for x in data["image_ids"]]
            self.text_ids = [str(x) for x in data["text_ids"]]
        expected_images = [str(x["id"]) for x in self.manifest["images"]]
        expected_texts = [str(x["id"]) for x in self.manifest["texts"]]
        if self.image_ids != expected_images or self.text_ids != expected_texts:
            raise ValueError("Feature row order does not exactly match the manifest.")
        if len(set(self.image_ids)) != len(self.image_ids) or len(set(self.text_ids)) != len(self.text_ids):
            raise ValueError("Manifest image and text IDs must be unique.")
        for name, tensor, count in (("image", self.images, len(self.image_ids)), ("text", self.texts, len(self.text_ids))):
            if tensor.shape != (count, dimension) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid {name} feature shape or nonfinite features.")
            norms = tensor.norm(dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4):
                raise ValueError(f"{name} features are not L2-normalized.")
        manifest_digest = sha256_file(self.manifest_path)
        declared_digest = self.metadata.get("manifest_sha256", self.metadata.get("input_manifest_sha256"))
        if declared_digest != manifest_digest:
            raise ValueError("Feature metadata must bind the exact manifest SHA256.")
        self.logit_scale = float(self.metadata["logit_scale"])
        if not math.isfinite(self.logit_scale) or self.logit_scale <= 0:
            raise ValueError("Invalid frozen logit scale.")
        image_lookup = {x: i for i, x in enumerate(self.image_ids)}
        text_lookup = {x: i for i, x in enumerate(self.text_ids)}
        self.pairs: list[dict[int, int]] = [dict() for _ in self.image_ids]
        for pair in self.manifest["pairs"]:
            image_index, text_index = image_lookup[pair["image_id"]], text_lookup[pair["text_id"]]
            code = RELATIONS[pair["relation"]]
            old = self.pairs[image_index].get(text_index)
            if old is not None and old != code:
                raise ValueError("Conflicting relation labels for the same ordered pair.")
            self.pairs[image_index][text_index] = code
        self.split_indices: dict[str, list[int]] = {}
        for index, image in enumerate(self.manifest["images"]):
            self.split_indices.setdefault(image["split"], []).append(index)
        for split in ("train", "validation"):
            if not self.split_indices.get(split):
                raise ValueError(f"Missing nonempty {split} split.")
            for index in self.split_indices[split]:
                if not any(code == 1 for code in self.pairs[index].values()):
                    raise ValueError(f"Image {self.image_ids[index]} lacks a source caption.")
        # A caption ID shared across train/validation/test would leak a candidate
        # across splits. Identical strings with distinct official IDs remain legal.
        text_splits: dict[int, set[str]] = {}
        for split, indices in self.split_indices.items():
            for image_index in indices:
                for text_index in self.pairs[image_index]:
                    text_splits.setdefault(text_index, set()).add(split)
        if any(len(splits) > 1 for splits in text_splits.values()):
            raise ValueError("A text ID is assigned to images across declared splits.")

    def batch(self, image_indices: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        selected = list(image_indices)
        # Sorting preserves the global manifest order and fixes retrieval ties.
        text_indices = sorted({j for i in selected for j in self.pairs[i]})
        lookup = {j: position for position, j in enumerate(text_indices)}
        relations = torch.zeros((len(selected), len(text_indices)), dtype=torch.int64)
        for row, image_index in enumerate(selected):
            for text_index, code in self.pairs[image_index].items():
                relations[row, lookup[text_index]] = code
        return self.images[selected], self.texts[text_indices], relations, text_indices

    def ledger_inputs(self) -> dict[str, Any]:
        def record(path: Path) -> dict[str, Any]:
            return {"path": str(path.relative_to(self.repository)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        return {
            "manifest": record(self.manifest_path), "features": record(self.features_path),
            "metadata": record(self.metadata_path), "feature_metadata": self.metadata,
            "training_image_ids": [self.image_ids[i] for i in self.split_indices["train"]],
            "validation_image_ids": [self.image_ids[i] for i in self.split_indices["validation"]],
            "training_text_ids": [self.text_ids[i] for i in sorted({j for k in self.split_indices["train"] for j in self.pairs[k]})],
            "validation_text_ids": [self.text_ids[i] for i in sorted({j for k in self.split_indices["validation"] for j in self.pairs[k]})],
            "logit_scale": self.logit_scale,
        }


@torch.inference_mode()
def validate_adapter(adapter: ResidualAdapter, data: FeatureDataset) -> dict[str, Any]:
    """Only the validation split contributes to fitting and checkpoint choice."""
    adapter.eval()
    image_indices = data.split_indices["validation"]
    images, texts, relations, text_indices = data.batch(image_indices)
    image_features, text_features = adapter(images, texts)
    # Evaluation accumulates dot products in float64 in both phases. The
    # positive frozen scale cannot change ranking and is deliberately omitted.
    scores = image_features.double() @ text_features.double().T
    winners = scores.argmax(dim=1)  # first index wins an exact tie
    positive = (relations == 1) | (relations == 2)
    retrieval = positive[torch.arange(len(image_indices)), winners].to(torch.float64)
    relation_values: list[float] = []
    by_image = []
    for row, image_index in enumerate(image_indices):
        support = scores[row, relations[row] == 2]
        contra = scores[row, relations[row] == 3]
        pair_count = support.numel() * contra.numel()
        relation_accuracy = None
        if pair_count:
            differences = support[:, None] - contra[None, :]
            relation_accuracy = float(((differences > 0).double() + 0.5 * (differences == 0).double()).mean())
            relation_values.append(relation_accuracy)
        by_image.append({
            "image_id": data.image_ids[image_index],
            "known_positive_i2t_r1": float(retrieval[row]),
            "winning_text_id": data.text_ids[text_indices[int(winners[row])]],
            "relation_accuracy": relation_accuracy,
            "supported_count": support.numel(), "contradicted_count": contra.numel(),
            "relation_pair_count": pair_count,
        })
    if not relation_values:
        raise ValueError("Validation has no supported-versus-contradicted comparisons.")
    r1 = float(retrieval.mean())
    relation_accuracy = float(np.mean(relation_values, dtype=np.float64))
    return {
        "known_positive_i2t_r1": r1, "relation_accuracy": relation_accuracy,
        "score": 0.5 * (r1 + relation_accuracy),
        "image_count": len(image_indices), "text_count": len(text_indices),
        "relation_eligible_images": len(relation_values), "per_image": by_image,
        "positive_relations": ["source", "supported"],
        "retrieval_tie_rule": "lowest_manifest_text_index", "relation_tie_credit": 0.5,
    }


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def assert_protocol_config(protocol: dict[str, Any], config: StudyConfig) -> None:
    """Bind every scientific setting to structured fields in the frozen plan.

    Torch thread count is an execution setting, still bound by the run ledger.
    New StudyConfig fields must be explicitly classified before fitting works.
    """
    fields = {
        "methods": ("training", "methods"),
        "learning_rates": ("training", "learning_rates"),
        "seeds": ("training", "seeds"),
        "epochs": ("training", "epochs"),
        "image_batch_size": ("training", "image_batch_size"),
        "feature_dim": ("encoder", "feature_dim"),
        "weight_decay": ("training", "weight_decay"),
        "adam_betas": ("training", "betas"),
        "adam_epsilon": ("training", "epsilon"),
        "adam_amsgrad": ("training", "amsgrad"),
        "adam_foreach": ("training", "foreach"),
        "adam_fused": ("training", "fused"),
        "grad_clip_norm": ("training", "gradient_clip_norm"),
        "optimizer": ("training", "optimizer"),
        "semantic_threshold": ("objectives", "fixed_threshold"),
        "semantic_softness": ("objectives", "semantic_softness"),
        "negative_weight": ("objectives", "constant_negative_weight"),
        "hardening_weight": ("objectives", "contradiction_weight"),
        "smoothing": ("objectives", "smoothing_coefficient"),
        "validation_positives": ("selection", "positive_relations"),
        "validation_tie_rule": ("selection", "retrieval_tie_rule"),
        "relation_tie_credit": ("selection", "hypothesis_ties"),
        "selection_tie_rule": ("selection", "tie_rule"),
        "device": ("training", "device"),
    }
    if set(fields) | {"threads"} != {field.name for field in dataclasses.fields(config)}:
        raise RuntimeError("Every StudyConfig field must have a protocol mapping or explicit runtime classification.")
    configured = json.loads(canonical_json(dataclasses.asdict(config)))
    mismatches = []
    for name, (section, key) in fields.items():
        if section not in protocol or key not in protocol[section]:
            raise ValueError(f"Frozen protocol lacks required structured field {section}.{key}.")
        expected = protocol[section][key]
        if configured[name] != expected:
            mismatches.append(f"{name}: configured {configured[name]!r}, protocol {expected!r}")
    if protocol.get("adapter", {}).get("dimension") != config.feature_dim:
        mismatches.append(f"feature_dim: adapter.dimension must also be {config.feature_dim}")
    if mismatches:
        raise ValueError("Frozen protocol disagrees with StudyConfig: " + "; ".join(mismatches))


def make_ledger(repository: Path, data: FeatureDataset, config: StudyConfig, protocol: Path, expected_protocol_sha256: str) -> dict[str, Any]:
    config.validate()
    protocol = protocol.resolve()
    actual = sha256_file(protocol)
    if actual != expected_protocol_sha256:
        raise ValueError("Frozen protocol hash mismatch; fitting is prohibited.")
    assert_protocol_config(json.loads(protocol.read_text()), config)
    source_paths = ["src/gcr/training.py", "src/gcr/adapters.py", "src/gcr/losses.py", "scripts/run_study.py"]
    sources = {name: sha256_file(repository / name) for name in source_paths}
    identity = {
        "schema_version": 1, "kind": "new_reconstructed_execution",
        "historical_results_read": False,
        "protocol": {"path": str(protocol.relative_to(repository)), "sha256": actual},
        "source_sha256": sources, "inputs": data.ledger_inputs(),
        "hyperparameters": dataclasses.asdict(config),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "selection": {
            "criterion": "mean_of_known_positive_i2t_r1_and_per_image_supported_contradicted_accuracy",
            "validation_positive_relations": ["source", "supported"],
            "candidate_best_epoch": "highest_score_then_earliest_epoch_including_epoch_zero",
            "method_learning_rate": "mean_seed_best_scores_then_lowest_learning_rate",
            "held_out_data_used": False,
        },
    }
    # Convert tuples to their JSON representation before future equality checks.
    identity = json.loads(canonical_json(identity))
    return {"identity": identity, "ledger_sha256": hashlib.sha256(canonical_json(identity)).hexdigest()}


def ensure_ledger(study_root: Path, ledger: dict[str, Any]) -> None:
    with file_lock(study_root / ".ledger.lock"):
        path = study_root / "ledger.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != ledger:
                raise ValueError("Study ledger is immutable. Changed inputs, protocol, code, or settings require a new output directory.")
        else:
            atomic_json(path, ledger)


def candidate_directory(study_root: Path, method: str, learning_rate: float, seed: int) -> Path:
    return study_root / "candidates" / method / f"lr_{learning_rate:.8g}" / f"seed_{seed}"


def _short_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in validation.items() if key != "per_image"}


def _completed_candidate(candidate_root: Path, ledger_sha256: str, method: str, learning_rate: float, seed: int, epochs: int) -> dict[str, Any] | None:
    completion_path = candidate_root / "completion.json"
    if not completion_path.exists():
        return None
    completion = json.loads(completion_path.read_text())
    expected = {"ledger_sha256": ledger_sha256, "method": method, "learning_rate": learning_rate, "seed": seed, "epochs_completed": epochs}
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible candidate completion record: {candidate_root}")
    for name in ("best.pt", "history.json"):
        path = candidate_root / name
        if not path.exists() or sha256_file(path) != completion["artifacts_sha256"].get(name):
            raise ValueError(f"Completed candidate artifact is missing or changed: {path}")
    return completion


def fit_candidate(repository: Path, study_root: Path, data: FeatureDataset, config: StudyConfig, ledger: dict[str, Any], method: str, learning_rate: float, seed: int) -> dict[str, Any]:
    if method not in config.methods or learning_rate not in config.learning_rates or seed not in config.seeds:
        raise ValueError("Requested candidate is outside the immutable study grid.")
    candidate_root = candidate_directory(study_root, method, learning_rate, seed)
    with file_lock(candidate_root / ".candidate.lock"):
        completed = _completed_candidate(candidate_root, ledger["ledger_sha256"], method, learning_rate, seed, config.epochs)
        if completed is not None:
            return {**completed, "skipped_compatible_complete": True}
        for name in ("history.json", "best.pt"):
            path = candidate_root / name
            if path.exists():
                # Partial executions can restart from epoch zero, but may never
                # silently reuse outputs from another protocol or code revision.
                old = json.loads(path.read_text()) if name.endswith("json") else torch.load(path, map_location="cpu", weights_only=True)
                if old.get("ledger_sha256") != ledger["ledger_sha256"]:
                    raise ValueError(f"Incompatible incomplete candidate artifact: {path}")
        started = time.monotonic()
        seed_everything(seed, config.threads)
        adapter = ResidualAdapter(config.feature_dim)
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=learning_rate, weight_decay=config.weight_decay,
            betas=config.adam_betas, eps=config.adam_epsilon,
            amsgrad=config.adam_amsgrad, foreach=config.adam_foreach,
            fused=config.adam_fused,
        )
        # Separate generators prevent method-specific random policies from
        # changing the image batch order used by the matched comparisons.
        order_generator = torch.Generator(device="cpu").manual_seed(seed)
        loss_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
        history: list[dict[str, Any]] = []
        best_score, best_epoch = -math.inf, -1
        training_indices = data.split_indices["train"]
        loss_kwargs = {
            "semantic_threshold": config.semantic_threshold,
            "semantic_softness": config.semantic_softness,
            "negative_weight": config.negative_weight,
            "hardening_weight": config.hardening_weight,
            "smoothing": config.smoothing,
        }
        for epoch in range(config.epochs + 1):
            epoch_started = time.monotonic()
            total_loss, total_images, gradient_steps = 0.0, 0, 0
            if epoch:
                adapter.train()
                order = torch.randperm(len(training_indices), generator=order_generator).tolist()
                for offset in range(0, len(order), config.image_batch_size):
                    indices = [training_indices[k] for k in order[offset:offset + config.image_batch_size]]
                    images, texts, relations, _ = data.batch(indices)
                    adapted_images, adapted_texts = adapter(images, texts)
                    loss = contrastive_loss(
                        adapted_images, adapted_texts, relations, method, data.logit_scale,
                        generator=loss_generator, semantic_text_features=texts,
                        semantic_image_features=images, **loss_kwargs,
                    )
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite or nonscalar loss in {method}, epoch {epoch}.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm, error_if_nonfinite=True)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient norm.")
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(indices)
                    total_images += len(indices)
                    gradient_steps += 1
            validation = validate_adapter(adapter, data)
            # Strict > means exact score ties retain the earlier epoch.
            improved = validation["score"] > best_score
            if improved:
                best_score, best_epoch = validation["score"], epoch
                atomic_torch_save(candidate_root / "best.pt", {
                    "schema_version": 1, "state_dict": adapter.state_dict(),
                    "ledger_sha256": ledger["ledger_sha256"], "method": method,
                    "learning_rate": learning_rate, "seed": seed, "epoch": epoch,
                    "validation": _short_validation(validation),
                    "hyperparameters": dataclasses.asdict(config),
                })
            history.append({
                "epoch": epoch, "mean_training_loss": total_loss / total_images if total_images else None,
                "training_images": total_images, "gradient_steps": gradient_steps,
                "validation": validation, "selected_best_so_far": improved,
                "seconds": time.monotonic() - epoch_started,
            })
            atomic_json(candidate_root / "history.json", {
                "schema_version": 1, "ledger_sha256": ledger["ledger_sha256"],
                "method": method, "learning_rate": learning_rate, "seed": seed,
                "best_epoch": best_epoch, "best_validation_score": best_score,
                "history": history,
            })
        completion = {
            "schema_version": 1, "ledger_sha256": ledger["ledger_sha256"],
            "method": method, "learning_rate": learning_rate, "seed": seed,
            "epochs_completed": config.epochs, "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "seconds": time.monotonic() - started,
            "artifacts_sha256": {name: sha256_file(candidate_root / name) for name in ("best.pt", "history.json")},
        }
        # This final atomic record is the only marker that authorizes skipping.
        atomic_json(candidate_root / "completion.json", completion)
        return completion


def ensure_epoch_zero(study_root: Path, data: FeatureDataset, config: StudyConfig, ledger: dict[str, Any]) -> None:
    with file_lock(study_root / ".epoch_zero.lock"):
        path = study_root / "epoch_zero.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("ledger_sha256") != ledger["ledger_sha256"] or payload.get("epoch") != 0:
                raise ValueError("Incompatible epoch-zero reference.")
            return
        seed_everything(config.seeds[0], config.threads)
        adapter = ResidualAdapter(config.feature_dim)
        validation = validate_adapter(adapter, data)
        atomic_torch_save(path, {
            "schema_version": 1, "state_dict": adapter.state_dict(),
            "ledger_sha256": ledger["ledger_sha256"], "method": "frozen",
            "learning_rate": 0.0, "seed": None, "epoch": 0,
            "validation": _short_validation(validation),
            "hyperparameters": dataclasses.asdict(config),
        })
        atomic_json(study_root / "epoch_zero_validation.json", validation)


def select_candidates(repository: Path, study_root: Path, config: StudyConfig, ledger: dict[str, Any]) -> dict[str, Any]:
    """Choose a shared learning rate per method using validation results only."""
    selection: dict[str, Any] = {
        "schema_version": 1, "ledger_sha256": ledger["ledger_sha256"],
        "held_out_data_used": False, "methods": {},
    }
    for method in config.methods:
        rates = []
        for learning_rate in config.learning_rates:
            candidates = []
            for seed in config.seeds:
                directory = candidate_directory(study_root, method, learning_rate, seed)
                result = _completed_candidate(directory, ledger["ledger_sha256"], method, learning_rate, seed, config.epochs)
                if result is None:
                    raise ValueError(f"Selection requires the complete declared grid; missing {directory}")
                candidates.append(result)
            rates.append({"learning_rate": learning_rate, "mean_validation_score": float(np.mean([x["best_validation_score"] for x in candidates])), "candidates": candidates})
        chosen = min(rates, key=lambda row: (-row["mean_validation_score"], row["learning_rate"]))
        runs = []
        for candidate in chosen["candidates"]:
            seed = candidate["seed"]
            source = candidate_directory(study_root, method, chosen["learning_rate"], seed) / "best.pt"
            destination = study_root / "selected" / method / f"seed_{seed}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            runs.append({
                "seed": seed, "epoch": payload["epoch"],
                "checkpoint": str(destination.relative_to(repository)),
                "checkpoint_sha256": sha256_file(destination),
                "candidate_checkpoint": str(source.relative_to(repository)),
                "validation": payload["validation"],
            })
        selection["methods"][method] = {
            "learning_rate": chosen["learning_rate"],
            "mean_validation_score": chosen["mean_validation_score"],
            "learning_rate_scores": [{"learning_rate": x["learning_rate"], "mean_validation_score": x["mean_validation_score"]} for x in rates],
            "runs": runs,
        }
    selection["selected_checkpoint_count"] = sum(len(x["runs"]) for x in selection["methods"].values())
    atomic_json(study_root / "selection.json", selection)
    return selection

