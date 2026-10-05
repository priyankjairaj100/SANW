"""Immutable, development-only execution of retention protocol v3.

This is a new execution reconstructed after the earlier uncommitted runtime was
lost. It reads no test predictions. All epochs are retained, exceeding the
protocol's minimum final/running-best storage requirement.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .adapters import ResidualAdapter
from .retention_losses import retention_loss
from .review_controls import apply_promotion_assignment, validate_assignment
from .review_training import (ASSIGNMENT_SEEDS, SourceRetrievalPool, _atomic_npz,
                              candidate_path, file_record, state_id, validate_source_retrieval)
from .training import (FeatureDataset, StudyConfig, atomic_json, atomic_torch_save,
                       canonical_json, file_lock, seed_everything, sha256_file, validate_adapter)

PROTOCOL_SHA256 = "5624342a20dd33b4f235264f57599a8bad0486d70fa587a6d9269371ea00b5d4"
POLICIES = ("source", "supported", "allocation_0.5", "allocation_0.8",
            "distilled_1", "distilled_4", "distilled_16", "image_source_only", "reverse_source_only")
FAMILIES = ("source", "supported", "allocation", "distilled", "wise_ft", "image_source_only", "reverse_source_only")
ALPHAS = (0., .1, .25, .5, .75, 1.)
CODE_PATHS = ("src/gcr/adapters.py", "src/gcr/losses.py", "src/gcr/training.py",
              "src/gcr/review_controls.py", "src/gcr/review_training.py",
              "src/gcr/retention_losses.py", "src/gcr/strengthen_retention.py",
              "scripts/run_strengthen_retention.py")


def original_config(repository: Path, dimension: int) -> tuple[dict, StudyConfig]:
    original = json.loads((repository / "results/study/ledger.json").read_text())
    identity = original["identity"]
    if hashlib.sha256(canonical_json(identity)).hexdigest() != original["ledger_sha256"]:
        raise ValueError("Original ledger content hash is invalid")
    for name, digest in identity["source_sha256"].items():
        if sha256_file(repository / name) != digest:
            raise ValueError(f"Frozen original implementation changed: {name}")
    hp = dict(identity["hyperparameters"])
    for name in ("methods", "learning_rates", "seeds", "adam_betas", "validation_positives"):
        hp[name] = tuple(hp[name])
    hp["feature_dim"] = dimension
    config = StudyConfig(**hp)
    config.validate()
    return original, config


def verify_protocol(path: Path, config: StudyConfig) -> dict:
    if sha256_file(path) != PROTOCOL_SHA256:
        raise ValueError("Frozen retention protocol SHA256 mismatch")
    protocol = json.loads(path.read_text())
    expected = {"policies": list(POLICIES), "learning_rates": list(config.learning_rates),
                "seeds": list(config.seeds), "epochs": config.epochs,
                "allocation_source_mix": [.5, .8], "distillation_beta": [1., 4., 16.],
                "distillation_temperature": 2., "wise_ft_alpha": list(ALPHAS), "fits_per_encoder": 81}
    if any(protocol["training"].get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration differs from the frozen retention protocol")
    if protocol["selection"]["families"] != list(FAMILIES):
        raise ValueError("Unexpected retention selection families")
    return protocol


def build_ledger(repository: Path, output: Path, encoder: str, config: StudyConfig,
                 data: FeatureDataset, pool: SourceRetrievalPool, protocol_path: Path,
                 assignment_directory: Path, assignment_parity_path: Path | None = None) -> tuple[dict, dict]:
    protocol = verify_protocol(protocol_path, config)
    if encoder not in ("vit_b32", "rn50") or config.feature_dim != {"vit_b32": 512, "rn50": 1024}[encoder]:
        raise ValueError("Encoder identifier and feature dimension disagree")
    original, inherited = original_config(repository, config.feature_dim)
    if config != inherited:
        raise ValueError("Complete original optimizer configuration must be inherited")
    base_path = repository / protocol["base"]
    base = json.loads(base_path.read_text())
    if hashlib.sha256(canonical_json(base["identity"])).hexdigest() != base["ledger_sha256"]:
        raise ValueError("Base review ledger hash is invalid")
    if data.manifest_path.resolve() != (repository / original["identity"]["inputs"]["manifest"]["path"]).resolve():
        raise ValueError("Training manifest must remain the original study manifest")
    if pool.manifest_path.resolve() != (repository / protocol["selection"]["retrieval_pool"]).resolve():
        raise ValueError("Development retrieval manifest must be the frozen 900-image pool")
    if sha256_file(data.manifest_path) != original["identity"]["inputs"]["manifest"]["sha256"]:
        raise ValueError("Training manifest content changed")
    input_recovery = {"status": "second_pretrained_encoder"}
    if encoder == "vit_b32":
        feature_digest = sha256_file(data.features_path)
        original_digest = original["identity"]["inputs"]["features"]["sha256"]
        original_metadata = original["identity"]["inputs"]["feature_metadata"]
        for key in ("model_revision", "weights_sha256", "open_clip_version", "logit_scale"):
            if data.metadata.get(key) != original_metadata.get(key) or key not in data.metadata:
                raise ValueError(f"Reconstructed ViT cache differs in pinned encoder provenance: {key}")
        input_recovery = {"status": "exact_original_cache" if feature_digest == original_digest else "reconstructed_cache_new_execution",
                          "original_features_sha256": original_digest, "actual_features_sha256": feature_digest,
                          "exact_original_replay_claimed": feature_digest == original_digest}
    if data.metadata.get("features_sha256") not in (None, sha256_file(data.features_path)):
        raise ValueError("Training feature metadata hash mismatch")
    assignments, assignment_records = {}, {}
    for draw, assignment_seed in enumerate(ASSIGNMENT_SEEDS):
        policy = f"score_stratified_draw_{draw}"
        path = assignment_directory / f"{policy}.json"
        record = json.loads(path.read_text())
        validate_assignment(record, expected_provenance={"manifest_sha256": sha256_file(data.manifest_path),
                            "features_sha256": sha256_file(data.features_path)},
                            expected_control="score_stratified", expected_seed=assignment_seed)
        if record["training_image_indices"] != sorted(data.split_indices["train"]):
            raise ValueError("Assignments must cover exactly the training split")
        if (record["global_image_count"], record["global_text_count"]) != (len(data.image_ids), len(data.text_ids)):
            raise ValueError("Assignment dimensions differ from training cache")
        assignments[draw] = record
        assignment_records[policy] = file_record(repository, path)
    parity_record = None
    if input_recovery["status"] == "reconstructed_cache_new_execution":
        parity_path = assignment_parity_path or assignment_directory.parent / "assignment_parity.json"
        receipt = json.loads(parity_path.read_text())
        if not receipt.get("passed") or receipt.get("actual_features_sha256") != sha256_file(data.features_path) or receipt.get("manifest_sha256") != sha256_file(data.manifest_path):
            raise ValueError("Reconstructed ViT input requires a passing input-bound control parity audit")
        for name, digest in receipt["source_sha256"].items():
            if sha256_file(repository / name) != digest:
                raise ValueError("Control recovery audit implementation changed")
        for policy, record in assignment_records.items():
            if receipt.get("regenerated_assignments", {}).get(policy) != record:
                raise ValueError("Reconstructed ViT controls differ from semantic-parity-audited files")
        parity_record = file_record(repository, parity_path)
    identity = {"schema_version": 1, "encoder": encoder, "architecture": "linear",
                "protocol": file_record(repository, protocol_path),
                "base_review_ledger": file_record(repository, base_path),
                "original_ledger_sha256": original["ledger_sha256"],
                "hyperparameters": dataclasses.asdict(config), "inputs": data.ledger_inputs(), "input_recovery": input_recovery,
                "development_retrieval": pool.ledger_inputs(), "assignments": assignment_records,
                "assignment_semantic_parity": parity_record,
                "source_sha256": {name: sha256_file(repository / name) for name in CODE_PATHS},
                "implementation": {"reconstruction": "new_execution_after_uncommitted_runtime_loss",
                    "teacher_features": "same_L2_normalization_as_zero_residual_initialization",
                    "kl_direction": "KL_teacher_to_student", "all_epoch_checkpoints_retained": True,
                    "wise_ft": "scale_both_residual_parameter_matrices_before_feature_normalization",
                    "test_outcomes_read": False},
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "platform": platform.platform()}}
    identity = json.loads(canonical_json(identity))
    ledger = {"identity": identity, "ledger_sha256": hashlib.sha256(canonical_json(identity)).hexdigest()}
    return ledger, assignments


def ensure_ledger(output: Path, ledger: dict) -> None:
    with file_lock(output / ".ledger.lock"):
        path = output / "ledger.json"
        if path.exists() and json.loads(path.read_text()) != ledger:
            raise ValueError("Retention ledger is immutable; changes require a new output directory")
        if not path.exists():
            atomic_json(path, ledger)


def update_norm(state: dict[str, torch.Tensor], alpha: float = 1.) -> float:
    return math.sqrt(sum(float((value.double() * alpha).square().sum()) for value in state.values()))


def scaled_adapter(adapter: ResidualAdapter, alpha: float) -> ResidualAdapter:
    if alpha not in ALPHAS:
        raise ValueError("WiSE interpolation coefficient outside the frozen grid")
    result = ResidualAdapter(adapter.image.weight.shape[0])
    result.load_state_dict({name: value.detach() * alpha for name, value in adapter.state_dict().items()})
    result.eval()
    return result


def _validate(adapter: ResidualAdapter, data: FeatureDataset, pool: SourceRetrievalPool,
              alphas: tuple[float, ...]) -> tuple[list[dict], dict]:
    variants, predictions = [], {}
    for alpha in alphas:
        model = adapter if alpha == 1. else scaled_adapter(adapter, alpha)
        native = validate_adapter(model, data)
        source, raw = validate_source_retrieval(model, pool)
        key = f"alpha_{alpha:.8g}"
        predictions.update({f"{key}__{name}": value for name, value in raw.items()})
        predictions[f"{key}__relation_image_ids"] = np.asarray([r["image_id"] for r in native["per_image"]])
        predictions[f"{key}__relation_accuracy"] = np.asarray([np.nan if r["relation_accuracy"] is None else r["relation_accuracy"] for r in native["per_image"]])
        variants.append({"alpha": alpha, "native": {k: v for k, v in native.items() if k != "per_image"},
                         "source_retrieval": source, "update_norm": update_norm(model.state_dict())})
    return variants, predictions


def _completed(directory: Path, ledger: dict, method: str, rate: float, seed: int,
               epochs: int, selection_sha256: str | None = None) -> dict | None:
    path = directory / "completion.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    expected = {"ledger_sha256": ledger["ledger_sha256"], "method": method,
                "learning_rate": rate, "seed": seed, "epochs_completed": epochs,
                "selection_sha256": selection_sha256, "checkpoint_count": epochs + 1}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Incompatible retention completion receipt")
    required = {"history.json"} | {f"epochs/epoch_{e:02d}.pt" for e in range(epochs + 1)} | {f"validation/epoch_{e:02d}.npz" for e in range(epochs + 1)}
    if set(receipt.get("artifact_sha256", {})) != required:
        raise ValueError("Completion must bind every epoch checkpoint and validation")
    for relative, digest in receipt["artifact_sha256"].items():
        if not (directory / relative).exists() or sha256_file(directory / relative) != digest:
            raise ValueError(f"Completed retention artifact missing or modified: {relative}")
    return receipt


def fit_candidate(repository: Path, output: Path, config: StudyConfig, data: FeatureDataset,
                  pool: SourceRetrievalPool, ledger: dict, method: str, rate: float, seed: int,
                  *, assignment: dict | None = None, target_epochs: int | None = None,
                  loss_policy: str | None = None, selection_sha256: str | None = None,
                  draw_id: int | None = None) -> dict:
    epochs = config.epochs if target_epochs is None else target_epochs
    control = method.startswith("matched_distilled_draw_")
    if rate not in config.learning_rates or seed not in config.seeds or not 0 <= epochs <= config.epochs:
        raise ValueError("Candidate schedule outside frozen protocol")
    if control:
        if assignment is None or draw_id not in range(3) or method != f"matched_distilled_draw_{draw_id}" or not selection_sha256 or loss_policy not in ("distilled_1", "distilled_4", "distilled_16"):
            raise ValueError("Matched control requires fixed draw, schedule, distillation and selection binding")
    elif method not in POLICIES or epochs != config.epochs or assignment is not None or loss_policy is not None:
        raise ValueError("Invalid full-grid retention candidate")
    effective_policy = loss_policy or method
    directory = candidate_path(output, method, rate, seed)
    with file_lock(directory / ".candidate.lock"):
        completed = _completed(directory, ledger, method, rate, seed, epochs, selection_sha256)
        if completed is not None:
            return {**completed, "skipped_compatible_complete": True}
        old_history = directory / "history.json"
        if old_history.exists() and json.loads(old_history.read_text()).get("ledger_sha256") != ledger["ledger_sha256"]:
            raise ValueError("Incompatible incomplete candidate")
        # Incomplete fits deterministically replay from initialization. Completed
        # fits are hash-verified and skipped. No unrecorded optimizer state used.
        started = time.monotonic()
        seed_everything(seed, config.threads)
        adapter = ResidualAdapter(config.feature_dim)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=rate, weight_decay=config.weight_decay,
            betas=config.adam_betas, eps=config.adam_epsilon, amsgrad=config.adam_amsgrad,
            foreach=config.adam_foreach, fused=config.adam_fused)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        training = data.split_indices["train"]
        history, hashes = [], {}
        for epoch in range(epochs + 1):
            epoch_start = time.monotonic()
            total, count, steps = 0., 0, 0
            if epoch:
                adapter.train()
                order = torch.randperm(len(training), generator=generator).tolist()
                for offset in range(0, len(order), config.image_batch_size):
                    indices = [training[i] for i in order[offset:offset + config.image_batch_size]]
                    images, texts, relations, text_indices = data.batch(indices)
                    if assignment is not None:
                        relations = apply_promotion_assignment(relations, indices, text_indices, assignment)
                    adapted_images, adapted_texts = adapter(images, texts)
                    loss = retention_loss(adapted_images, adapted_texts, relations, effective_policy,
                        data.logit_scale, frozen_images=images, frozen_texts=texts)
                    if loss.ndim or not torch.isfinite(loss):
                        raise FloatingPointError("Invalid retention loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip_norm, error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                    count += len(indices)
                    steps += 1
            variants, predictions = _validate(adapter, data, pool, ALPHAS if method == "supported" else (1.,))
            checkpoint_rel = f"epochs/epoch_{epoch:02d}.pt"
            predictions_rel = f"validation/epoch_{epoch:02d}.npz"
            sid = f"{ledger['identity']['encoder']}__{state_id(method, rate, seed, epoch)}"
            checkpoint = {"schema_version": 1, "state_dict": adapter.state_dict(), "state_id": sid,
                "encoder": ledger["identity"]["encoder"], "architecture": "linear", "method": method,
                "learning_rate": rate, "seed": seed, "epoch": epoch, "draw_id": draw_id,
                "loss_policy": effective_policy, "ledger_sha256": ledger["ledger_sha256"],
                "protocol_sha256": PROTOCOL_SHA256, "selection_sha256": selection_sha256,
                "base_hyperparameters": dataclasses.asdict(config), "update_norm": update_norm(adapter.state_dict())}
            atomic_torch_save(directory / checkpoint_rel, checkpoint)
            _atomic_npz(directory / predictions_rel, **predictions)
            for relative in (checkpoint_rel, predictions_rel):
                hashes[relative] = sha256_file(directory / relative)
            history.append({"epoch": epoch, "state_id": sid, "mean_training_loss": total / count if count else None,
                "training_images": count, "gradient_steps": steps, "validation": variants,
                "checkpoint": checkpoint_rel, "checkpoint_sha256": hashes[checkpoint_rel],
                "source_retrieval_predictions": predictions_rel, "seconds": time.monotonic() - epoch_start})
            atomic_json(directory / "history.json", {"schema_version": 1, "ledger_sha256": ledger["ledger_sha256"],
                "method": method, "loss_policy": effective_policy, "learning_rate": rate, "seed": seed,
                "draw_id": draw_id, "selection_sha256": selection_sha256, "history": history})
        hashes["history.json"] = sha256_file(directory / "history.json")
        receipt = {"schema_version": 1, "ledger_sha256": ledger["ledger_sha256"], "method": method,
            "learning_rate": rate, "seed": seed, "epochs_completed": epochs, "checkpoint_count": epochs + 1,
            "selection_sha256": selection_sha256, "artifact_sha256": hashes,
            "draw_id": draw_id, "loss_policy": effective_policy, "seconds": time.monotonic() - started}
        atomic_json(directory / "completion.json", receipt)
        return receipt


def feasible(validation: dict, frozen: dict, tolerance_pp: float) -> bool:
    return all(validation["source_retrieval"][key] >= frozen["source_retrieval"][key] - tolerance_pp / 100 - 1e-12
               for key in ("i2t_r1", "t2i_r1"))


def choose_family(candidates: list[dict], seeds: tuple[int, ...], tolerance_pp: float) -> dict:
    """Pure selector: candidates share one method parameter/rate across seeds.

    Each candidate has learning_rate, parameter, and histories {seed: variants
    with epoch}. Test data is neither accepted nor accessed by this function.
    """
    choices = []
    for candidate in candidates:
        best, feasible_epochs = {}, {}
        if set(candidate["histories"]) != set(seeds):
            raise ValueError("Each candidate must contain all declared seeds")
        for seed in seeds:
            history = candidate["histories"][seed]
            if [row["epoch"] for row in history] != list(range(len(history))):
                raise ValueError("Selector requires consecutive epochs including zero")
            for row in history:
                values = (row["native"]["relation_accuracy"], row["source_retrieval"]["i2t_r1"], row["source_retrieval"]["t2i_r1"])
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise ValueError("Development accuracies must be finite probabilities")
            eligible = [row for row in history if feasible(row, history[0], tolerance_pp)]
            feasible_epochs[seed] = [row["epoch"] for row in eligible]
            best[seed] = min(eligible, key=lambda row: (-row["native"]["relation_accuracy"], row["epoch"]))
        choices.append({"learning_rate": candidate["learning_rate"], "parameter": candidate["parameter"],
            "policy": candidate["policy"], "alpha": candidate.get("alpha", 1.), "best": best,
            "feasible_epochs": feasible_epochs,
            "mean_relation_accuracy": float(np.mean([row["native"]["relation_accuracy"] for row in best.values()], dtype=np.float64))})
    if not choices:
        raise ValueError("No family candidates")
    chosen = min(choices, key=lambda row: (-row["mean_relation_accuracy"], row["learning_rate"], row["parameter"]))
    return {**chosen, "candidate_scores": [{k: v for k, v in row.items() if k != "best"} for row in choices]}


def _state(repository: Path, directory: Path, ledger: dict, policy: str, rate: float,
           seed: int, row: dict, *, family: str | None = None, alpha: float = 1., draw_id=None) -> dict:
    default_family = policy.split("_")[0] if policy.startswith(("allocation_", "distilled_")) else policy
    return {"state_id": row["state_id"], "encoder": ledger["identity"]["encoder"], "architecture": "linear",
            "method": policy, "family": family or default_family,
            "seed": seed, "epoch": row["epoch"], "learning_rate": rate,
            "checkpoint": str((directory / row["checkpoint"]).resolve().relative_to(repository.resolve())),
            "checkpoint_sha256": row["checkpoint_sha256"], "alpha": alpha,
            "beta": float(policy.split("_")[1]) if policy.startswith("distilled_") else None,
            "source_mix": float(policy.split("_")[1]) if policy.startswith("allocation_") else None,
            "draw_id": draw_id, "update_norm": next(v["update_norm"] for v in row["validation"] if v["alpha"] == alpha)}


def select_and_export(repository: Path, output: Path, config: StudyConfig, ledger: dict) -> dict:
    histories, states = {}, []
    frozen_validation = None
    for policy in POLICIES:
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_path(output, policy, rate, seed)
                if _completed(directory, ledger, policy, rate, seed, config.epochs) is None:
                    raise ValueError(f"Selection requires all 81 complete candidates: {directory}")
                document = json.loads((directory / "history.json").read_text())
                if any(document.get(key) != value for key, value in {"ledger_sha256": ledger["ledger_sha256"],
                       "method": policy, "learning_rate": rate, "seed": seed}.items()):
                    raise ValueError("Retention history metadata mismatch")
                rows = document["history"]
                if [row["epoch"] for row in rows] != list(range(config.epochs + 1)):
                    raise ValueError("History must contain every epoch zero through ten")
                for row in rows:
                    expected_alphas = ALPHAS if policy == "supported" else (1.,)
                    if [v["alpha"] for v in row["validation"]] != list(expected_alphas):
                        raise ValueError("History misses declared interpolation variants")
                    if sha256_file(directory / row["checkpoint"]) != row["checkpoint_sha256"]:
                        raise ValueError("Checkpoint history binding mismatch")
                    if row["epoch"] == 0:
                        for v in row["validation"]:
                            identity = {k: v[k] for k in ("native", "source_retrieval", "update_norm")}
                            if frozen_validation is None:
                                frozen_validation = identity
                            if identity != frozen_validation or v["update_norm"] != 0.:
                                raise ValueError("Epoch-zero outputs differ across training/interpolation candidates")
                    states.append(_state(repository, directory, ledger, policy, rate, seed, row))
                histories[(policy, rate, seed)] = rows
    selections = []
    for tolerance_pp, name in ((1., "primary"), (0., "sensitivity")):
        result = {"schema_version": 1, "ledger_sha256": ledger["ledger_sha256"],
            "protocol_sha256": PROTOCOL_SHA256, "encoder": ledger["identity"]["encoder"],
            "tolerance_pp": tolerance_pp, "families": {}, "test_outcomes_used": False}
        for family in FAMILIES:
            options = []
            policies = ("supported",) if family == "wise_ft" else tuple(p for p in POLICIES if p == family or p.startswith(family + "_"))
            for policy in policies:
                for rate in config.learning_rates:
                    for alpha in ALPHAS if family == "wise_ft" else (1.,):
                        parameter = alpha if family == "wise_ft" else float(policy.split("_")[1]) if family in ("allocation", "distilled") else 0.
                        seed_histories = {seed: [{**next(v for v in row["validation"] if v["alpha"] == alpha), "epoch": row["epoch"], "state_id": row["state_id"]}
                                                  for row in histories[(policy, rate, seed)]] for seed in config.seeds}
                        options.append({"policy": policy, "learning_rate": rate, "parameter": parameter,
                                        "alpha": alpha, "histories": seed_histories})
            chosen = choose_family(options, config.seeds, tolerance_pp)
            policy, rate, alpha = chosen["policy"], chosen["learning_rate"], chosen["alpha"]
            runs = []
            for seed, best in chosen["best"].items():
                row = histories[(policy, rate, seed)][best["epoch"]]
                directory = candidate_path(output, policy, rate, seed)
                state = _state(repository, directory, ledger, policy, rate, seed, row, family=family, alpha=alpha)
                if family == "wise_ft":
                    source_checkpoint = torch.load(repository / state["checkpoint"], map_location="cpu", weights_only=True)
                    scaled = {key: value * alpha for key, value in source_checkpoint["state_dict"].items()}
                    sid = f"{state['state_id']}__wise_alpha_{alpha:.8g}"
                    path = output / "selected" / f"{sid}.pt"
                    expected_checkpoint = {**source_checkpoint, "state_dict": scaled, "state_id": sid,
                        "method": "wise_ft", "alpha": alpha, "parameters_already_scaled": True,
                        "source_checkpoint_sha256": state["checkpoint_sha256"], "update_norm": update_norm(scaled)}
                    if not path.exists():
                        atomic_torch_save(path, expected_checkpoint)
                    else:
                        previous = torch.load(path, map_location="cpu", weights_only=True)
                        if previous.get("source_checkpoint_sha256") != state["checkpoint_sha256"] or previous.get("alpha") != alpha or any(not torch.equal(scaled[k], previous["state_dict"][k]) for k in scaled):
                            raise ValueError("Previously exported interpolation checkpoint changed")
                    state = {**state, "state_id": sid, "method": "wise_ft", "checkpoint": str(path.relative_to(repository)),
                        "checkpoint_sha256": sha256_file(path), "parameters_already_scaled": True}
                    if not any(s["state_id"] == sid for s in states):
                        states.append(state)
                run = {**state, "validation": {k: best[k] for k in ("native", "source_retrieval")},
                       "nonzero_learned": best["epoch"] > 0 and best["update_norm"] > 0.}
                runs.append(run)
                selections.append({"state_id": state["state_id"], "family": family, "seed": seed, "tolerance_pp": tolerance_pp})
            result["families"][family] = {k: v for k, v in chosen.items() if k != "best"}
            result["families"][family].update({"runs": runs, "all_three_nonzero": len(runs) == 3 and all(r["nonzero_learned"] for r in runs)})
        path = output / f"selection_{name}.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(canonical_json(result)):
            raise ValueError("An existing development selection is immutable")
        if not path.exists():
            atomic_json(path, result)
    manifest = {"schema_version": 1, "ledger_sha256": ledger["ledger_sha256"], "protocol_sha256": PROTOCOL_SHA256,
        "encoder": ledger["identity"]["encoder"], "architecture": "linear", "logit_scale": ledger["identity"]["inputs"]["logit_scale"], "states": states, "selections": selections,
        "test_outcomes_used_for_selection": False, "matched_controls_complete": False,
        "selection_sha256": {name: sha256_file(output / f"selection_{name}.json") for name in ("primary", "sensitivity")}}
    manifest_path = output / "state_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("matched_controls_complete"):
        # Re-selection must never remove subsequently completed matched controls.
        previous = json.loads(manifest_path.read_text())
        if previous["selection_sha256"] != manifest["selection_sha256"]:
            raise ValueError("Completed controls bind a different selection")
        return previous
    atomic_json(manifest_path, manifest)
    return manifest


def fit_matched_controls(repository: Path, output: Path, config: StudyConfig, data: FeatureDataset,
                         pool: SourceRetrievalPool, ledger: dict, assignments: dict) -> dict:
    selection_path = output / "selection_primary.json"
    selected = json.loads(selection_path.read_text())
    manifest = json.loads((output / "state_manifest.json").read_text())
    if selected["ledger_sha256"] != ledger["ledger_sha256"] or manifest["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Matched controls require this ledger's frozen selection")
    selection_hash = sha256_file(selection_path)
    if manifest["selection_sha256"]["primary"] != selection_hash:
        raise ValueError("Selection changed before matched controls")
    support = selected["families"]["distilled"]
    rate, beta = support["learning_rate"], support["parameter"]
    loss_policy = f"distilled_{beta:.8g}"
    for run in support["runs"]:
        seed, epoch = run["seed"], run["epoch"]
        for draw in range(3):
            method = f"matched_distilled_draw_{draw}"
            fit_candidate(repository, output, config, data, pool, ledger, method, rate, seed,
                assignment=assignments[draw], target_epochs=epoch, loss_policy=loss_policy,
                selection_sha256=selection_hash, draw_id=draw)
            directory = candidate_path(output, method, rate, seed)
            row = json.loads((directory / "history.json").read_text())["history"][-1]
            state = _state(repository, directory, ledger, method, rate, seed, row,
                           family="matched_distilled", draw_id=draw)
            state["beta"] = beta
            state["assignment_seed"] = ASSIGNMENT_SEEDS[draw]
            state["selection_sha256"] = selection_hash
            if not any(r["state_id"] == state["state_id"] for r in manifest["states"]):
                manifest["states"].append(state)
                manifest["selections"].append({"state_id": state["state_id"], "family": "matched_distilled",
                    "seed": seed, "draw_id": draw, "tolerance_pp": 1.})
    manifest["matched_controls_complete"] = True
    atomic_json(output / "state_manifest.json", manifest)
    return manifest

