"""Review-extension fitting with immutable assignments and every epoch retained.

This module does not modify the original experiment or use test outcomes.
The native score reuses the frozen original validation implementation. A second
selector uses source-only retrieval on the declared development pool.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np
import torch

from .adapters import ResidualAdapter
from .losses import contrastive_loss
from .training import (
    FeatureDataset, StudyConfig, atomic_json, atomic_torch_save, canonical_json,
    file_lock, seed_everything, sha256_file, validate_adapter,
)


ASSIGNMENT_SEEDS = (101, 211, 307)
POLICIES = (
    "source", "supported",
    "count_only_draw_0", "count_only_draw_1", "count_only_draw_2",
    "score_stratified_draw_0", "score_stratified_draw_1", "score_stratified_draw_2",
)


def policy_parts(policy: str) -> tuple[str, int | None, int | None]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown review policy: {policy}")
    if policy in ("source", "supported"):
        return policy, None, None
    condition, draw = policy.rsplit("_draw_", 1)
    return condition, int(draw), ASSIGNMENT_SEEDS[int(draw)]


def state_id(policy: str, rate: float, seed: int, epoch: int) -> str:
    return f"{policy}__lr_{rate:.8g}__seed_{seed}__epoch_{epoch:02d}"


def candidate_path(output: Path, policy: str, rate: float, seed: int) -> Path:
    return output / "candidates" / policy / f"lr_{rate:.8g}" / f"seed_{seed}"


def file_record(repository: Path, path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve().relative_to(repository.resolve())),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class SourceRetrievalPool:
    """Source-caption-only development pool with explicit ownership relevance."""

    def __init__(self, repository: Path, manifest: Path, features: Path, metadata: Path, training_data: FeatureDataset, dimension: int = 512):
        self.repository, self.manifest_path = repository.resolve(), manifest.resolve()
        self.features_path, self.metadata_path = features.resolve(), metadata.resolve()
        self.manifest = json.loads(manifest.read_text())
        self.metadata = json.loads(metadata.read_text())
        self.image_ids = [str(row["id"]) for row in self.manifest["images"]]
        self.text_ids = [str(row["id"]) for row in self.manifest["texts"]]
        if len(set(self.image_ids)) != len(self.image_ids) or len(set(self.text_ids)) != len(self.text_ids):
            raise ValueError("Development retrieval IDs must be unique.")
        if any(row["split"] != "validation" for row in self.manifest["images"]):
            raise ValueError("The source retrieval selector may read only a validation pool.")
        excluded = {training_data.image_ids[i] for split in ("train", "calibration", "test") for i in training_data.split_indices.get(split, [])}
        if excluded.intersection(self.image_ids):
            raise ValueError("Source retrieval development images overlap training, calibration, or held-out test images.")
        if len(self.image_ids) != 900 or len(self.text_ids) != 4500:
            raise ValueError("The frozen review selector requires exactly 900 development images and 4500 captions.")
        if self.metadata.get("manifest_sha256", self.metadata.get("input_manifest_sha256")) != sha256_file(manifest):
            raise ValueError("Development feature metadata does not bind the input manifest.")
        declared_features = self.metadata.get("features_sha256")
        if declared_features is not None and declared_features != sha256_file(features):
            raise ValueError("Development feature digest differs from metadata.")
        with np.load(features, allow_pickle=False) as data:
            if self.image_ids != data["image_ids"].astype(str).tolist() or self.text_ids != data["text_ids"].astype(str).tolist():
                raise ValueError("Development feature IDs do not match manifest order.")
            self.images = torch.from_numpy(np.array(data["image_features"], dtype=np.float32, copy=True))
            self.texts = torch.from_numpy(np.array(data["text_features"], dtype=np.float32, copy=True))
        for values, count in ((self.images, len(self.image_ids)), (self.texts, len(self.text_ids))):
            if values.shape != (count, dimension) or not torch.isfinite(values).all():
                raise ValueError("Invalid source-retrieval feature values or shape.")
            if not torch.allclose(values.norm(dim=1), torch.ones(count), atol=2e-4, rtol=2e-4):
                raise ValueError("Source retrieval features must be normalized.")
        for key in ("model_revision", "weights_sha256", "open_clip_version"):
            if key not in self.metadata or self.metadata[key] != training_data.metadata.get(key):
                raise ValueError(f"Source retrieval encoder provenance differs: {key}")
        image_lookup = {name: i for i, name in enumerate(self.image_ids)}
        text_lookup = {name: i for i, name in enumerate(self.text_ids)}
        owner = [-1] * len(self.text_ids)
        for pair in self.manifest["pairs"]:
            if pair["relation"] != "source":
                raise ValueError("Source retrieval selection must use only original source captions.")
            i, j = image_lookup[pair["image_id"]], text_lookup[pair["text_id"]]
            if owner[j] not in (-1, i):
                raise ValueError("A development caption has multiple source owners.")
            owner[j] = i
        if any(value < 0 for value in owner) or np.bincount(owner, minlength=len(self.image_ids)).tolist() != [5] * len(self.image_ids):
            raise ValueError("Source retrieval requires exactly five owned captions for every image.")
        self.owner = torch.tensor(owner, dtype=torch.int64)

    def ledger_inputs(self) -> dict[str, Any]:
        return {"manifest": file_record(self.repository, self.manifest_path),
                "features": file_record(self.repository, self.features_path),
                "metadata": file_record(self.repository, self.metadata_path),
                "image_ids": self.image_ids, "text_ids": self.text_ids,
                "relevance": "source_caption_ownership", "split": "validation"}


@torch.inference_mode()
def validate_source_retrieval(adapter: ResidualAdapter, pool: SourceRetrievalPool) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    adapter.eval()
    images, texts = adapter(pool.images, pool.texts)
    scores = images.double() @ texts.double().T
    image_winners = scores.argmax(dim=1)
    text_winners = scores.argmax(dim=0)
    image_correct = pool.owner[image_winners] == torch.arange(len(pool.image_ids))
    text_correct = text_winners == pool.owner
    i2t, t2i = float(image_correct.double().mean()), float(text_correct.double().mean())
    summary = {"i2t_r1": i2t, "t2i_r1": t2i, "score": (i2t + t2i) / 2,
               "image_count": len(pool.image_ids), "text_count": len(pool.text_ids),
               "relevance": "source_caption_ownership",
               "tie_rule": "first_candidate_in_manifest_order", "score_accumulation": "float64"}
    predictions = {"image_ids": np.asarray(pool.image_ids), "text_ids": np.asarray(pool.text_ids),
                   "image_correct": image_correct.numpy(), "text_correct": text_correct.numpy(),
                   "image_winning_text_index": image_winners.numpy(), "text_winning_image_index": text_winners.numpy()}
    return summary, predictions


def load_original_context(repository: Path) -> tuple[dict[str, Any], StudyConfig, FeatureDataset]:
    original = json.loads((repository / "results/study/ledger.json").read_text())
    identity = original["identity"]
    if hashlib.sha256(canonical_json(identity)).hexdigest() != original["ledger_sha256"]:
        raise ValueError("The original study ledger is invalid.")
    for name, digest in identity["source_sha256"].items():
        if sha256_file(repository / name) != digest:
            raise ValueError(f"The frozen original study source changed: {name}")
    if sha256_file(repository / identity["protocol"]["path"]) != identity["protocol"]["sha256"]:
        raise ValueError("The frozen original study protocol changed.")
    for kind in ("manifest", "features", "metadata"):
        item = identity["inputs"][kind]
        if sha256_file(repository / item["path"]) != item["sha256"]:
            raise ValueError(f"The frozen original {kind} input changed.")
    hp = dict(identity["hyperparameters"])
    for name in ("methods", "learning_rates", "seeds", "adam_betas", "validation_positives"):
        hp[name] = tuple(hp[name])
    config = StudyConfig(**hp)
    config.validate()
    inputs = identity["inputs"]
    data = FeatureDataset(repository, repository / inputs["manifest"]["path"], repository / inputs["features"]["path"], repository / inputs["metadata"]["path"], config.feature_dim)
    return original, config, data


def verify_protocol(path: Path, expected_sha256: str, original: dict[str, Any], config: StudyConfig) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Review follow-up protocol SHA256 mismatch; fitting is prohibited.")
    protocol = json.loads(path.read_text())
    expected = {
        ("training", "policies"): list(POLICIES),
        ("training", "learning_rates"): list(config.learning_rates),
        ("training", "seeds"): list(config.seeds),
        ("training", "epochs"): config.epochs,
        ("training", "image_batch_size"): config.image_batch_size,
        ("training", "optimizer"): config.optimizer,
        ("training", "weight_decay"): config.weight_decay,
        ("training", "betas"): list(config.adam_betas),
        ("training", "epsilon"): config.adam_epsilon,
        ("training", "gradient_clip_norm"): config.grad_clip_norm,
        ("training", "threads"): config.threads,
        ("training", "device"): config.device,
        ("training", "amsgrad"): config.adam_amsgrad,
        ("training", "foreach"): config.adam_foreach,
        ("training", "fused"): config.adam_fused,
        ("training", "scheduler"): "none",
        ("training", "feature_dim"): config.feature_dim,
        ("training", "inherit_exact_base_hyperparameters"): True,
        ("training", "assignment_seeds"): list(ASSIGNMENT_SEEDS),
        ("training", "candidate_count"): 72,
        ("training", "epoch_records"): 792,
        ("training", "primary_epoch"): 10,
        ("training", "trajectory_epochs"): [1, 5, 10],
        ("controls", "assignment_seeds"): list(ASSIGNMENT_SEEDS),
        ("controls", "assignment_draws"): 3,
        ("evaluation", "primary_epoch"): 10,
        ("evaluation", "trajectory_epochs"): [1, 5, 10],
        ("selection", "criteria"): ["native", "source_retrieval"],
        ("selection", "native_code"): "original_mean_known_positive_i2t_and_relation_accuracy",
        ("selection", "source_retrieval_code"): "mean_i2t_t2i_r1",
        ("selection", "epoch_zero_eligible"): True,
        ("selection", "epoch_ties"): "earliest",
        ("selection", "learning_rate_ties"): "lower",
        ("selection", "random_draw_selection"): "separate_no_winner_draw",
    }
    errors = []
    if protocol.get("base", {}).get("ledger_identity_sha256") != original["ledger_sha256"]:
        errors.append("base.ledger_identity_sha256")
    if dataclasses.asdict(config) != original["identity"]["hyperparameters"]:
        # JSON serialization converts tuple-valued hyperparameters to lists.
        if json.loads(canonical_json(dataclasses.asdict(config))) != original["identity"]["hyperparameters"]:
            errors.append("complete inherited base hyperparameters")
    for (section, key), value in expected.items():
        if protocol.get(section, {}).get(key) != value:
            errors.append(f"{section}.{key} must equal {value!r}")
    if errors:
        raise ValueError("Review protocol disagrees with the execution: " + "; ".join(errors))
    return protocol


def assignment_paths(output: Path) -> dict[str, Path]:
    return {policy: output / "assignments" / f"{policy}.json" for policy in POLICIES if policy not in ("source", "supported")}


def prepare_assignments(repository: Path, output: Path, protocol_path: Path, protocol_sha256: str, original: dict[str, Any], config: StudyConfig, data: FeatureDataset) -> dict[str, Any]:
    from .review_controls import make_promotion_assignment, save_assignment
    verify_protocol(protocol_path, protocol_sha256, original, config)
    provenance = {"manifest_sha256": sha256_file(data.manifest_path),
                  "features_sha256": sha256_file(data.features_path),
                  "protocol_sha256": protocol_sha256}
    records = {}
    for policy, path in assignment_paths(output).items():
        condition, _, assignment_seed = policy_parts(policy)
        record = make_promotion_assignment(
            data.image_ids, data.text_ids, data.images, data.texts, data.pairs,
            training_image_indices=data.split_indices["train"], control=condition,
            assignment_seed=assignment_seed, provenance=provenance,
            text_strings=[row["text"] for row in data.manifest["texts"]],
        )
        save_assignment(path, record)
        records[policy] = file_record(repository, path)
    atomic_json(output / "assignment_files.json", {"protocol_sha256": protocol_sha256, "assignments": records})
    return records


def build_review_ledger(repository: Path, output: Path, protocol_path: Path, protocol_sha256: str, original: dict[str, Any], config: StudyConfig, data: FeatureDataset, pool: SourceRetrievalPool) -> tuple[dict[str, Any], dict[str, Any]]:
    from .review_controls import load_assignment
    protocol = verify_protocol(protocol_path, protocol_sha256, original, config)
    for name, actual in (("manifest", pool.manifest_path), ("features", pool.features_path)):
        if actual.resolve() != (repository / protocol["data"]["source_validation"][name]).resolve():
            raise ValueError(f"Source retrieval {name} is not the frozen protocol path.")
    for field, digest_field in (("protocol", "protocol_sha256"), ("selection", "selection_sha256")):
        if sha256_file(repository / protocol["base"][field]) != protocol["base"][digest_field]:
            raise ValueError(f"The original {field} differs from the follow-up protocol.")
    assignment_receipt = json.loads((output / "assignment_files.json").read_text())
    if assignment_receipt.get("protocol_sha256") != protocol_sha256 or set(assignment_receipt.get("assignments", {})) != set(assignment_paths(output)):
        raise ValueError("Assignment-file receipt does not bind all six controls to this protocol.")
    provenance = {"manifest_sha256": sha256_file(data.manifest_path),
                  "features_sha256": sha256_file(data.features_path),
                  "protocol_sha256": protocol_sha256}
    assignments, records = {}, {}
    for policy, path in assignment_paths(output).items():
        if not path.exists():
            raise ValueError(f"Freeze all six assignment files before fitting; missing {path}")
        condition, _, assignment_seed = policy_parts(policy)
        record = load_assignment(path, expected_file_sha256=assignment_receipt["assignments"][policy]["sha256"],
                                 expected_provenance=provenance, expected_control=condition,
                                 expected_seed=assignment_seed)
        if record["source_sha256"] != sha256_file(repository / "src/gcr/review_controls.py"):
            raise ValueError("Assignment was generated with a different control implementation.")
        if record["training_image_indices"] != sorted(data.split_indices["train"]):
            raise ValueError("Assignment is not defined on exactly the original training images.")
        if record["global_image_count"] != len(data.image_ids) or record["global_text_count"] != len(data.text_ids):
            raise ValueError("Assignment global indices do not match the training manifest.")
        assignments[policy] = record
        records[policy] = file_record(repository, path)
    code = ["src/gcr/review_training.py", "scripts/run_review_followup.py", "src/gcr/review_controls.py", "src/gcr/training.py", "src/gcr/adapters.py", "src/gcr/losses.py"]
    identity = {
        "schema_version": 1, "kind": "post_hoc_review_followup", "test_outcomes_used_for_selection": False,
        "protocol": file_record(repository, protocol_path), "protocol_contents": protocol,
        "base_study": {"ledger_sha256": original["ledger_sha256"], **file_record(repository, repository / "results/study/ledger.json")},
        "training_inputs": data.ledger_inputs(), "source_retrieval_validation": pool.ledger_inputs(),
        "base_hyperparameters": dataclasses.asdict(config), "policies": list(POLICIES),
        "assignment_files": records,
        "assignment_receipt": file_record(repository, output / "assignment_files.json"),
        "source_sha256": {path: sha256_file(repository / path) for path in code},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform()},
    }
    identity = json.loads(canonical_json(identity))
    ledger = {"identity": identity, "ledger_sha256": hashlib.sha256(canonical_json(identity)).hexdigest()}
    return ledger, assignments


def ensure_review_ledger(output: Path, ledger: dict[str, Any]) -> None:
    with file_lock(output / ".ledger.lock"):
        path = output / "ledger.json"
        if path.exists() and json.loads(path.read_text()) != ledger:
            raise ValueError("Review ledger is immutable; changed code, protocol, assignments, inputs, or runtime requires a new output directory.")
        if not path.exists():
            atomic_json(path, ledger)


def _completed(directory: Path, ledger: dict[str, Any], policy: str, rate: float, seed: int) -> dict[str, Any] | None:
    path = directory / "completion.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    expected = {"ledger_sha256": ledger["ledger_sha256"], "method": policy, "learning_rate": rate, "seed": seed, "epochs_completed": 10}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible review completion: {directory}")
    required = {"history.json"} | {f"epochs/epoch_{epoch:02d}.pt" for epoch in range(11)} | {f"validation/epoch_{epoch:02d}.npz" for epoch in range(11)}
    if set(record.get("artifact_sha256", {})) != required or record.get("checkpoint_count") != 11:
        raise ValueError("A completed review candidate must bind all eleven checkpoints and validations.")
    for relative, digest in record["artifact_sha256"].items():
        path = directory / relative
        if not path.exists() or sha256_file(path) != digest:
            raise ValueError(f"A completed review artifact is missing or modified: {path}")
    return record


def _metric_only(validation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in validation.items() if key != "per_image"}


def _original_reference(repository: Path, policy: str, rate: float, seed: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if policy not in ("source", "supported"):
        return None
    original_method = "clip" if policy == "source" else "multipositive"
    directory = repository / "results/study/candidates" / original_method / f"lr_{rate:.8g}" / f"seed_{seed}"
    receipt = json.loads((directory / "completion.json").read_text())
    for name in ("history.json", "best.pt"):
        if sha256_file(directory / name) != receipt["artifacts_sha256"][name]:
            raise ValueError("Original replay reference artifact was modified.")
    return json.loads((directory / "history.json").read_text()), torch.load(directory / "best.pt", map_location="cpu", weights_only=True)


def fit_review_candidate(repository: Path, output: Path, config: StudyConfig, data: FeatureDataset, pool: SourceRetrievalPool, ledger: dict[str, Any], assignments: dict[str, Any], policy: str, rate: float, seed: int) -> dict[str, Any]:
    from .review_controls import apply_promotion_assignment
    if policy not in POLICIES or rate not in config.learning_rates or seed not in config.seeds:
        raise ValueError("Candidate is outside the frozen follow-up grid.")
    if policy not in ("source", "supported") and policy not in assignments:
        raise ValueError("Randomized policy requires its frozen promotion assignment.")
    directory = candidate_path(output, policy, rate, seed)
    with file_lock(directory / ".candidate.lock"):
        completed = _completed(directory, ledger, policy, rate, seed)
        if completed is not None:
            return {**completed, "skipped_compatible_complete": True}
        history_path = directory / "history.json"
        if history_path.exists() and json.loads(history_path.read_text()).get("ledger_sha256") != ledger["ledger_sha256"]:
            raise ValueError("Incompatible incomplete review candidate.")
        for path in (directory / "epochs").glob("*.pt"):
            if torch.load(path, map_location="cpu", weights_only=True).get("ledger_sha256") != ledger["ledger_sha256"]:
                raise ValueError("Incompatible incomplete epoch checkpoint.")
        started = time.monotonic()
        seed_everything(seed, config.threads)
        adapter = ResidualAdapter(config.feature_dim)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=rate, weight_decay=config.weight_decay,
                                      betas=config.adam_betas, eps=config.adam_epsilon,
                                      amsgrad=config.adam_amsgrad, foreach=config.adam_foreach, fused=config.adam_fused)
        order_generator = torch.Generator(device="cpu").manual_seed(seed)
        loss_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
        condition, draw_id, assignment_seed = policy_parts(policy)
        original_reference = _original_reference(repository, policy, rate, seed)
        parity = {"required": original_reference is not None, "checked_epochs": 0,
                  "native_validation_exact": True, "training_loss_exact": True,
                  "best_state_exact": None, "max_best_parameter_error": 0.0}
        training_indices = data.split_indices["train"]
        history, hashes = [], {}
        for epoch in range(config.epochs + 1):
            epoch_started = time.monotonic()
            total_loss, total_images, gradient_steps = 0.0, 0, 0
            if epoch:
                adapter.train()
                order = torch.randperm(len(training_indices), generator=order_generator).tolist()
                for offset in range(0, len(order), config.image_batch_size):
                    indices = [training_indices[k] for k in order[offset:offset + config.image_batch_size]]
                    images, texts, relations, text_indices = data.batch(indices)
                    if policy in assignments:
                        relations = apply_promotion_assignment(relations, indices, text_indices, assignments[policy])
                    adapted_images, adapted_texts = adapter(images, texts)
                    method = "clip" if policy == "source" else "multipositive"
                    loss = contrastive_loss(adapted_images, adapted_texts, relations, method, data.logit_scale,
                                            generator=loss_generator, semantic_text_features=texts,
                                            semantic_image_features=images, semantic_threshold=config.semantic_threshold,
                                            semantic_softness=config.semantic_softness, negative_weight=config.negative_weight,
                                            hardening_weight=config.hardening_weight, smoothing=config.smoothing)
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise FloatingPointError("Invalid follow-up training loss.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm, error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(indices)
                    total_images += len(indices)
                    gradient_steps += 1
            native = validate_adapter(adapter, data)
            source_retrieval, prediction = validate_source_retrieval(adapter, pool)
            training_loss = total_loss / total_images if total_images else None
            if original_reference is not None:
                old_history, old_checkpoint = original_reference
                old_epoch = old_history["history"][epoch]
                parity["checked_epochs"] += 1
                parity["native_validation_exact"] &= native == old_epoch["validation"]
                parity["training_loss_exact"] &= training_loss == old_epoch["mean_training_loss"]
                if epoch == old_checkpoint["epoch"]:
                    parity["best_state_exact"] = all(torch.equal(value, old_checkpoint["state_dict"][name]) for name, value in adapter.state_dict().items())
                    parity["max_best_parameter_error"] = max(float((value - old_checkpoint["state_dict"][name]).abs().max()) for name, value in adapter.state_dict().items())
                if not parity["native_validation_exact"] or not parity["training_loss_exact"] or parity["best_state_exact"] is False:
                    atomic_json(directory / "replay_parity_failure.json", {**parity, "epoch": epoch})
                    raise RuntimeError(f"Follow-up replay differs from the frozen original candidate: {policy}, rate {rate}, seed {seed}, epoch {epoch}")
            checkpoint_relative = f"epochs/epoch_{epoch:02d}.pt"
            predictions_relative = f"validation/epoch_{epoch:02d}.npz"
            atomic_torch_save(directory / checkpoint_relative, {
                "schema_version": 1, "state_dict": adapter.state_dict(), "ledger_sha256": ledger["ledger_sha256"],
                "protocol_sha256": ledger["identity"]["protocol"]["sha256"], "method": policy,
                "condition": condition, "draw_id": draw_id, "assignment_seed": assignment_seed,
                "learning_rate": rate, "seed": seed, "epoch": epoch,
                "state_id": state_id(policy, rate, seed, epoch),
                "validation": {"native": _metric_only(native), "source_retrieval": source_retrieval},
                "base_hyperparameters": dataclasses.asdict(config),
            })
            _atomic_npz(directory / predictions_relative, **prediction)
            hashes[checkpoint_relative] = sha256_file(directory / checkpoint_relative)
            hashes[predictions_relative] = sha256_file(directory / predictions_relative)
            history.append({"epoch": epoch, "mean_training_loss": training_loss,
                            "training_images": total_images, "gradient_steps": gradient_steps,
                            "validation": {"native": native, "source_retrieval": source_retrieval},
                            "source_retrieval_predictions": predictions_relative,
                            "checkpoint": checkpoint_relative, "checkpoint_sha256": hashes[checkpoint_relative],
                            "seconds": time.monotonic() - epoch_started})
            atomic_json(history_path, {"schema_version": 1, "ledger_sha256": ledger["ledger_sha256"],
                                       "method": policy, "condition": condition, "draw_id": draw_id,
                                       "learning_rate": rate, "seed": seed, "history": history, "replay_parity": parity})
        hashes["history.json"] = sha256_file(history_path)
        completed = {"schema_version": 1, "ledger_sha256": ledger["ledger_sha256"], "method": policy,
                     "condition": condition, "draw_id": draw_id, "assignment_seed": assignment_seed,
                     "learning_rate": rate, "seed": seed, "epochs_completed": config.epochs,
                     "checkpoint_count": config.epochs + 1, "artifact_sha256": hashes,
                     "replay_parity": parity, "seconds": time.monotonic() - started}
        atomic_json(directory / "completion.json", completed)
        return completed


def select_and_export(repository: Path, output: Path, config: StudyConfig, ledger: dict[str, Any]) -> dict[str, Any]:
    histories = {}
    for policy in POLICIES:
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_path(output, policy, rate, seed)
                if _completed(directory, ledger, policy, rate, seed) is None:
                    raise ValueError(f"Selection requires all 72 completed candidates; missing {directory}")
                candidate_history = json.loads((directory / "history.json").read_text())
                if any(candidate_history.get(key) != value for key, value in {"ledger_sha256": ledger["ledger_sha256"], "method": policy, "learning_rate": rate, "seed": seed}.items()):
                    raise ValueError("Review history metadata does not match its candidate.")
                rows = candidate_history["history"]
                if [row["epoch"] for row in rows] != list(range(11)):
                    raise ValueError("Review history must contain each epoch 0 through 10 exactly once.")
                for row in rows:
                    if row["checkpoint"] != f"epochs/epoch_{row['epoch']:02d}.pt" or sha256_file(directory / row["checkpoint"]) != row["checkpoint_sha256"]:
                        raise ValueError("Review history checkpoint binding is invalid.")
                    for selector in ("native", "source_retrieval"):
                        if not np.isfinite(row["validation"][selector]["score"]):
                            raise ValueError("Selection scores must be finite.")
                histories[(policy, rate, seed)] = rows
    states, selections = [], []
    for (policy, rate, seed), history in histories.items():
        condition, draw, assignment_seed = policy_parts(policy)
        directory = candidate_path(output, policy, rate, seed)
        for row in history:
            states.append({"state_id": state_id(policy, rate, seed, row["epoch"]),
                           "condition": condition, "method": policy, "draw_id": draw,
                           "assignment_seed": assignment_seed, "learning_rate": rate, "seed": seed,
                           "epoch": row["epoch"], "checkpoint": str((directory / row["checkpoint"]).relative_to(repository)),
                           "checkpoint_sha256": row["checkpoint_sha256"],
                           "validation": {key: _metric_only(value) for key, value in row["validation"].items()}})
    if len(states) != 792 or len({row["state_id"] for row in states}) != 792:
        raise ValueError("The follow-up manifest requires exactly 792 unique states.")
    for selector in ("native", "source_retrieval"):
        selected = {"schema_version": 1, "selector": selector, "ledger_sha256": ledger["ledger_sha256"],
                    "protocol_sha256": ledger["identity"]["protocol"]["sha256"], "methods": {},
                    "test_outcomes_used": False, "draw_selection": "separate_no_winner_draw"}
        for policy in POLICIES:
            condition, draw, assignment_seed = policy_parts(policy)
            rate_rows = []
            for rate in config.learning_rates:
                best = {seed: min(histories[(policy, rate, seed)], key=lambda row: (-row["validation"][selector]["score"], row["epoch"])) for seed in config.seeds}
                rate_rows.append({"learning_rate": rate,
                                  "mean_validation_score": float(np.mean([row["validation"][selector]["score"] for row in best.values()])),
                                  "best": best})
            chosen = min(rate_rows, key=lambda row: (-row["mean_validation_score"], row["learning_rate"]))
            runs = []
            for seed, row in chosen["best"].items():
                selection = {"selector": selector, "condition": condition, "method": policy,
                             "draw_id": draw, "assignment_seed": assignment_seed,
                             "learning_rate": chosen["learning_rate"], "seed": seed, "epoch": row["epoch"],
                             "state_id": state_id(policy, chosen["learning_rate"], seed, row["epoch"])}
                selections.append(selection)
                runs.append({**selection, "validation": _metric_only(row["validation"][selector])})
            selected["methods"][policy] = {"condition": condition, "draw_id": draw,
                                            "assignment_seed": assignment_seed, "learning_rate": chosen["learning_rate"],
                                            "mean_validation_score": chosen["mean_validation_score"],
                                            "learning_rate_scores": [{k: v for k, v in row.items() if k != "best"} for row in rate_rows],
                                            "runs": runs}
        if selector == "native":
            original_selection = json.loads((repository / "results/study/selection.json").read_text())
            for policy, original_method in (("source", "clip"), ("supported", "multipositive")):
                new, old = selected["methods"][policy], original_selection["methods"][original_method]
                if new["learning_rate"] != old["learning_rate"] or {r["seed"]: r["epoch"] for r in new["runs"]} != {r["seed"]: r["epoch"] for r in old["runs"]}:
                    raise RuntimeError("Native source/supported selection differs from original despite required replay parity.")
        atomic_json(output / f"selection_{selector}.json", selected)
    manifest = {"schema_version": 1, "protocol_sha256": ledger["identity"]["protocol"]["sha256"],
                "ledger_sha256": ledger["ledger_sha256"], "states": states, "selections": selections,
                "candidate_count": len(histories), "state_count": len(states),
                "primary_epoch": 10, "trajectory_epochs": [1, 5, 10],
                "test_outcomes_used_for_selection": False, "draw_aggregation": "aggregate_all_three_draws_without_winner_selection"}
    atomic_json(output / "state_manifest.json", manifest)
    return manifest
