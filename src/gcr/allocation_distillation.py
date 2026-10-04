"""Exploratory A+D extension, separate from the completed retention protocol.

The extension preserves all candidate captions and inherits the original
optimizer and development selector. It cannot claim fresh confirmatory tests:
the same held-out benchmarks already informed the preceding research program.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import ResidualAdapter
from .retention_losses import retention_loss, source_distribution_kl
from .review_controls import apply_promotion_assignment, make_promotion_assignment, validate_assignment
from .review_training import ASSIGNMENT_SEEDS, SourceRetrievalPool, _atomic_npz, candidate_path, file_record, state_id
from .strengthen_retention import (
    ALPHAS, CODE_PATHS as BASE_CODE_PATHS, PROTOCOL_SHA256 as BASE_PROTOCOL_SHA256,
    _completed, _validate, choose_family, ensure_ledger, original_config, update_norm,
)
from .training import (
    FeatureDataset, StudyConfig, atomic_json, atomic_torch_save, canonical_json,
    file_lock, seed_everything, sha256_file,
)

SOURCE_MIXES = (.5, .8)
BETAS = (1., 4., 16.)
JOINT_POLICIES = tuple(f"allocation_distillation_{mix:g}_{beta:g}"
                       for mix in SOURCE_MIXES for beta in BETAS)
BASELINE_POLICIES = ("source", "supported", "allocation_0.5", "allocation_0.8",
                     "distilled_1", "distilled_4", "distilled_16")
POLICIES = BASELINE_POLICIES + JOINT_POLICIES
FAMILIES = ("source", "supported", "allocation", "distilled", "allocation_distillation", "wise_ft")
CODE_PATHS = BASE_CODE_PATHS + ("src/gcr/allocation_distillation.py",
                              "src/gcr/allocation_distillation_analysis.py", "src/gcr/evaluation.py",
                              "scripts/run_allocation_distillation.py")
EXTENSION_ID = "allocation_distillation_v1"
RN50_PROTOCOL_PATH = "docs/STRENGTHEN_ENCODER_REPLICATION_PROTOCOL.json"
RN50_PROTOCOL_SHA256 = "53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e"


def allocation_distillation_loss(
    images: Tensor, texts: Tensor, relations: Tensor, logit_scale: float, *,
    source_mix: float, beta: float, frozen_images: Tensor | None = None,
    frozen_texts: Tensor | None = None, temperature: float = 2.,
) -> Tensor:
    """Return lambda L_source + (1-lambda) L_supported + beta T² source KL.

    The source KL is the equally weighted mean of the two original directions.
    Boundary values support exact objective and gradient identity checks.
    Actual fitting accepts only the protocol's two mixtures and three betas.
    """
    if not math.isfinite(source_mix) or not 0 <= source_mix <= 1:
        raise ValueError("source_mix must be finite and between zero and one")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if temperature != 2.:
        raise ValueError("The extension uses temperature two")
    if source_mix == 0 and beta in BETAS:
        return retention_loss(images, texts, relations, f"distilled_{beta:g}", logit_scale,
                              frozen_images=frozen_images, frozen_texts=frozen_texts,
                              temperature=temperature)
    if source_mix == 0:
        supervised = retention_loss(images, texts, relations, "supported", logit_scale)
    elif source_mix == 1:
        supervised = retention_loss(images, texts, relations, "source", logit_scale)
    else:
        supervised = (source_mix * retention_loss(images, texts, relations, "source", logit_scale)
                      + (1 - source_mix) * retention_loss(images, texts, relations, "supported", logit_scale))
    if beta == 0:
        return supervised
    if frozen_images is None or frozen_texts is None:
        raise ValueError("Source KL requires frozen teacher features")
    scale = torch.as_tensor(logit_scale, dtype=images.dtype, device=images.device).detach()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("logit_scale must be a finite positive scalar")
    student = (images @ texts.T) * scale
    with torch.no_grad():
        teacher = (F.normalize(frozen_images.detach(), dim=-1)
                   @ F.normalize(frozen_texts.detach(), dim=-1).T) * scale
    image_kl, text_kl = source_distribution_kl(student, teacher, relations, temperature)
    return supervised + beta * temperature ** 2 * .5 * (image_kl + text_kl)


def policy_parameters(policy: str) -> tuple[str, float, float]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown extension policy: {policy}")
    if policy in JOINT_POLICIES:
        mix, beta = map(float, policy.removeprefix("allocation_distillation_").split("_"))
        return "allocation_distillation", mix, beta
    if policy.startswith("allocation_"):
        return "allocation", float(policy.removeprefix("allocation_")), 0.
    if policy.startswith("distilled_"):
        return "distilled", 0., float(policy.removeprefix("distilled_"))
    return policy, float(policy == "source"), 0.


def policy_loss(images, texts, relations, policy, logit_scale, *, frozen_images, frozen_texts):
    if policy not in JOINT_POLICIES:
        if policy not in BASELINE_POLICIES:
            raise ValueError("Policy is outside the extension grid")
        return retention_loss(images, texts, relations, policy, logit_scale,
                              frozen_images=frozen_images, frozen_texts=frozen_texts)
    _, mix, beta = policy_parameters(policy)
    return allocation_distillation_loss(images, texts, relations, logit_scale,
        source_mix=mix, beta=beta, frozen_images=frozen_images, frozen_texts=frozen_texts)


def protocol_template() -> dict:
    """Return a draft specification. Writing it does not authorize fitting."""
    from .allocation_distillation_analysis import evaluation_specification
    return {
        "schema_version": 1, "extension_id": EXTENSION_ID,
        "status": "exploratory_after_completed_v3",
        "base_protocol": {"path": "results/strengthen_retention/protocol_v3.json",
                          "sha256": BASE_PROTOCOL_SHA256},
        "training": {"policies": list(POLICIES), "learning_rates": [1e-4, 3e-4, 1e-3],
                     "seeds": [17, 29, 43], "epochs": 10, "source_mix": list(SOURCE_MIXES),
                     "beta": list(BETAS), "temperature": 2., "fits_per_encoder": 117,
                     "joint_fits_per_encoder": 54, "other_settings": "exact_original_StudyConfig",
                     "objective": "lambda L_source + (1-lambda) L_supported + beta T^2 (KL_image + KL_text)/2",
                     "candidate_contents": "unchanged source captions and hypotheses",
                     "teacher_domain": "source-caption columns; each reverse row covers all batch images",
                     "checkpoints": "every epoch, including zero",
                     "wise_ft_alpha": list(ALPHAS),
                     "matched_controls": {"fits_per_encoder": 9, "assignment_seeds": list(ASSIGNMENT_SEEDS),
                         "policy": "score_stratified", "selection": "none",
                         "schedule": "copy primary A+D source_mix, beta, learning rate, and per-seed epoch",
                         "aggregation": "mean three assignment draws within each seed, then mean seeds"}},
        "selection": {"families": list(FAMILIES), "primary_tolerance_pp": 1.,
                      "sensitivity_tolerance_pp": 0., "relation_pool": "original 100-image validation split",
                      "retrieval_pool": "data/review_followup/e_vil_dev900/manifest.json",
                      "rule": "unchanged retention-v3 choose_family and feasible functions",
                      "shared_ties": "lower learning rate, then lower source_mix and beta for A+D; lower scalar parameter otherwise",
                      "test_used": False, "freeze_before_test": True},
        "evaluation": evaluation_specification(),
    }


def verify_protocol(repository: Path, path: Path, expected_sha256: str, config: StudyConfig) -> dict:
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError("Extension protocol SHA256 mismatch")
    protocol = json.loads(path.read_text())
    template = protocol_template()
    for field in ("schema_version", "extension_id", "status", "base_protocol", "training", "selection"):
        if protocol.get(field) != template[field]:
            raise ValueError(f"Extension protocol disagrees with implemented {field}")
    parent = repository / protocol["base_protocol"]["path"]
    if sha256_file(parent) != BASE_PROTOCOL_SHA256:
        raise ValueError("The parent retention protocol changed")
    if (list(config.learning_rates), list(config.seeds), config.epochs) != ([1e-4, 3e-4, 1e-3], [17, 29, 43], 10):
        raise ValueError("Extension schedule differs from the complete original grid")
    if protocol.get("evaluation") != template["evaluation"]:
        raise ValueError("Freeze the exact exploratory evaluation specification before fitting")
    return protocol


def generate_assignments(repository: Path, output: Path, data: FeatureDataset,
                         protocol_sha256: str) -> dict:
    """Generate the existing two-bin controls with the extension protocol bound."""
    provenance = {"manifest_sha256": sha256_file(data.manifest_path),
                  "features_sha256": sha256_file(data.features_path), "protocol_sha256": protocol_sha256}
    records = {}
    for draw, seed in enumerate(ASSIGNMENT_SEEDS):
        policy = f"score_stratified_draw_{draw}"
        record = make_promotion_assignment(data.image_ids, data.text_ids, data.images, data.texts,
            data.pairs, data.split_indices["train"], "score_stratified", seed, provenance,
            text_strings=[row["text"] for row in data.manifest["texts"]])
        path = output / f"{policy}.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("An existing extension assignment is immutable")
        if not path.exists():
            atomic_json(path, record)
        records[policy] = file_record(repository, path)
    return records


def build_ledger(repository: Path, output: Path, encoder: str, config: StudyConfig,
                 data: FeatureDataset, pool: SourceRetrievalPool, protocol_path: Path,
                 protocol_sha256: str, assignment_directory: Path) -> tuple[dict, dict]:
    protocol = verify_protocol(repository, protocol_path, protocol_sha256, config)
    if not output.resolve().is_relative_to((repository / "results/allocation_distillation").resolve()):
        raise ValueError("Extension output must stay under results/allocation_distillation")
    if encoder not in ("vit_b32", "rn50") or config.feature_dim != {"vit_b32": 512, "rn50": 1024}[encoder]:
        raise ValueError("Encoder identifier and feature dimension disagree")
    original, inherited = original_config(repository, config.feature_dim)
    if config != inherited:
        raise ValueError("The extension must inherit the complete original optimizer configuration")
    original_inputs = original["identity"]["inputs"]
    if (data.manifest_path.resolve() != (repository / original_inputs["manifest"]["path"]).resolve()
            or sha256_file(data.manifest_path) != original_inputs["manifest"]["sha256"]):
        raise ValueError("The extension must use the unchanged original training manifest")
    if pool.manifest_path.resolve() != (repository / protocol["selection"]["retrieval_pool"]).resolve():
        raise ValueError("The extension must use the original development retrieval manifest")
    parent = json.loads((repository / protocol["base_protocol"]["path"]).read_text())
    review_path = repository / parent["base"]
    review = json.loads(review_path.read_text())
    if hashlib.sha256(canonical_json(review["identity"])).hexdigest() != review["ledger_sha256"]:
        raise ValueError("The original review ledger has an invalid hash")
    if sha256_file(pool.manifest_path) != review["identity"]["source_retrieval_validation"]["manifest"]["sha256"]:
        raise ValueError("The development retrieval manifest changed")
    metadata_hash = data.metadata.get("features_sha256")
    actual_features_hash = sha256_file(data.features_path)
    if metadata_hash not in (None, actual_features_hash):
        raise ValueError("Training cache and metadata hash disagree")
    if encoder == "vit_b32":
        prior_metadata = original_inputs["feature_metadata"]
        for key in ("model_revision", "weights_sha256", "open_clip_version", "logit_scale"):
            if key not in data.metadata or data.metadata[key] != prior_metadata.get(key):
                raise ValueError(f"The ViT encoder provenance changed: {key}")
    else:
        rn50_path = repository / RN50_PROTOCOL_PATH
        if sha256_file(rn50_path) != RN50_PROTOCOL_SHA256:
            raise ValueError("The original RN50 replication protocol changed")
        pinned = json.loads(rn50_path.read_text())["encoder"]
        for key in ("weights_sha256", "dimension"):
            if data.metadata.get(key) != pinned[key]:
                raise ValueError(f"The RN50 encoder provenance changed: {key}")
        if data.metadata.get("features_sha256") != actual_features_hash:
            raise ValueError("RN50 metadata must bind the exact feature cache")
    assignments, assignment_records = {}, {}
    for draw, assignment_seed in enumerate(ASSIGNMENT_SEEDS):
        policy = f"score_stratified_draw_{draw}"
        path = assignment_directory / f"{policy}.json"
        record = json.loads(path.read_text())
        validate_assignment(record, expected_provenance={"manifest_sha256": sha256_file(data.manifest_path),
                            "features_sha256": actual_features_hash, "protocol_sha256": protocol_sha256},
                            expected_control="score_stratified", expected_seed=assignment_seed)
        if record["training_image_indices"] != sorted(data.split_indices["train"]):
            raise ValueError("Assignments must cover the exact training split")
        if (record["global_image_count"], record["global_text_count"]) != (len(data.image_ids), len(data.text_ids)):
            raise ValueError("Assignment dimensions differ from the training cache")
        assignments[draw] = record
        assignment_records[policy] = file_record(repository, path)
    identity = {
        "schema_version": 1, "extension_id": EXTENSION_ID, "encoder": encoder,
        "architecture": "linear", "protocol": file_record(repository, protocol_path),
        "base_review_ledger": file_record(repository, review_path),
        "original_ledger_sha256": original["ledger_sha256"],
        "hyperparameters": dataclasses.asdict(config), "inputs": data.ledger_inputs(),
        "development_retrieval": pool.ledger_inputs(),
        "assignments": assignment_records,
        "encoder_protocol": file_record(repository, repository / RN50_PROTOCOL_PATH) if encoder == "rn50" else None,
        "source_sha256": {name: sha256_file(repository / name) for name in CODE_PATHS},
        "implementation": {"study_status": "exploratory_after_test_exposure",
                           "baseline_execution": "fresh_same_cache_and_full_grid",
                           "all_epoch_checkpoints_retained": True, "selection_uses_test": False,
                           "parent_files_modified": False},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "platform": platform.platform()},
    }
    identity = json.loads(canonical_json(identity))
    return {"identity": identity, "ledger_sha256": hashlib.sha256(canonical_json(identity)).hexdigest()}, assignments


def fit_candidate(repository: Path, output: Path, config: StudyConfig, data: FeatureDataset,
                  pool: SourceRetrievalPool, ledger: dict, method: str, rate: float, seed: int,
                  *, assignment: dict | None = None, target_epochs: int | None = None,
                  loss_policy: str | None = None, selection_sha256: str | None = None,
                  draw_id: int | None = None) -> dict:
    """Use the v3 training order and optimizer; only the joint objective is new."""
    epochs = config.epochs if target_epochs is None else target_epochs
    control = method.startswith("matched_allocation_distillation_draw_")
    if rate not in config.learning_rates or seed not in config.seeds or not 0 <= epochs <= config.epochs:
        raise ValueError("Candidate lies outside the extension grid")
    if control:
        if (assignment is None or draw_id not in range(3) or
            method != f"matched_allocation_distillation_draw_{draw_id}" or not selection_sha256 or
            loss_policy not in JOINT_POLICIES):
            raise ValueError("Matched control requires a fixed assignment, joint policy, and selection binding")
    elif (method not in POLICIES or epochs != config.epochs or assignment is not None or
          loss_policy is not None or selection_sha256 is not None or draw_id is not None):
        raise ValueError("Invalid full-grid extension candidate")
    effective_policy = loss_policy or method
    protocol_sha256 = ledger["identity"]["protocol"]["sha256"]
    directory = candidate_path(output, method, rate, seed)
    with file_lock(directory / ".candidate.lock"):
        completed = _completed(directory, ledger, method, rate, seed, epochs, selection_sha256)
        if completed is not None:
            return {**completed, "skipped_compatible_complete": True}
        old_history = directory / "history.json"
        if old_history.exists() and json.loads(old_history.read_text()).get("ledger_sha256") != ledger["ledger_sha256"]:
            raise ValueError("An incomplete candidate belongs to another ledger")
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
                    loss = policy_loss(adapted_images, adapted_texts, relations, effective_policy, data.logit_scale,
                                       frozen_images=images, frozen_texts=texts)
                    if loss.ndim or not torch.isfinite(loss):
                        raise FloatingPointError("Invalid allocation-distillation loss")
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
            family, mix, beta = policy_parameters(effective_policy)
            checkpoint = {"schema_version": 1, "extension_id": EXTENSION_ID,
                "state_dict": adapter.state_dict(), "state_id": sid,
                "encoder": ledger["identity"]["encoder"], "architecture": "linear", "method": method,
                "family": "matched_allocation_distillation" if control else family,
                "source_mix": mix, "beta": beta, "draw_id": draw_id,
                "learning_rate": rate, "seed": seed, "epoch": epoch,
                "loss_policy": effective_policy, "ledger_sha256": ledger["ledger_sha256"],
                "protocol_sha256": protocol_sha256, "selection_sha256": selection_sha256,
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
            "learning_rate": rate, "seed": seed, "epochs_completed": epochs,
            "checkpoint_count": epochs + 1, "selection_sha256": selection_sha256,
            "artifact_sha256": hashes, "loss_policy": effective_policy, "draw_id": draw_id,
            "seconds": time.monotonic() - started}
        atomic_json(directory / "completion.json", receipt)
        return receipt


def _state(repository, output, ledger, policy, rate, seed, row, *, alpha=1., family=None,
           loss_policy=None, draw_id=None):
    policy_family, mix, beta = policy_parameters(loss_policy or policy)
    path = candidate_path(output, policy, rate, seed) / row["checkpoint"]
    return {"state_id": row["state_id"], "encoder": ledger["identity"]["encoder"],
            "architecture": "linear", "method": policy, "family": family or policy_family,
            "seed": seed, "epoch": row["epoch"], "learning_rate": rate,
            "checkpoint": str(path.resolve().relative_to(repository.resolve())),
            "checkpoint_sha256": row["checkpoint_sha256"], "alpha": alpha,
            "beta": beta, "source_mix": mix, "draw_id": draw_id,
            "update_norm": next(v["update_norm"] for v in row["validation"] if v["alpha"] == alpha)}


def select_and_export(repository: Path, output: Path, config: StudyConfig, ledger: dict) -> dict:
    histories, states = {}, []
    frozen = None
    for policy in POLICIES:
        for rate in config.learning_rates:
            for seed in config.seeds:
                directory = candidate_path(output, policy, rate, seed)
                if _completed(directory, ledger, policy, rate, seed, config.epochs) is None:
                    raise ValueError(f"Selection requires every extension candidate: {directory}")
                document = json.loads((directory / "history.json").read_text())
                expected = {"ledger_sha256": ledger["ledger_sha256"], "method": policy,
                            "learning_rate": rate, "seed": seed}
                if any(document.get(key) != value for key, value in expected.items()):
                    raise ValueError("Extension history metadata mismatch")
                rows = document["history"]
                if [row["epoch"] for row in rows] != list(range(config.epochs + 1)):
                    raise ValueError("A candidate history must contain every epoch")
                for row in rows:
                    expected_alphas = ALPHAS if policy == "supported" else (1.,)
                    if [variant["alpha"] for variant in row["validation"]] != list(expected_alphas):
                        raise ValueError("Unexpected extension validation variants")
                    if sha256_file(directory / row["checkpoint"]) != row["checkpoint_sha256"]:
                        raise ValueError("Checkpoint history binding mismatch")
                    if row["epoch"] == 0:
                        for value in row["validation"]:
                            identity = {key: value[key] for key in ("native", "source_retrieval", "update_norm")}
                            frozen = identity if frozen is None else frozen
                            if identity != frozen or value["update_norm"] != 0.:
                                raise ValueError("Epoch-zero validation differs between candidates")
                    states.append(_state(repository, output, ledger, policy, rate, seed, row))
                histories[(policy, rate, seed)] = rows
    selections = []
    protocol_hash = ledger["identity"]["protocol"]["sha256"]
    for tolerance, name in ((1., "primary"), (0., "sensitivity")):
        result = {"schema_version": 1, "extension_id": EXTENSION_ID,
                  "ledger_sha256": ledger["ledger_sha256"], "protocol_sha256": protocol_hash,
                  "encoder": ledger["identity"]["encoder"], "tolerance_pp": tolerance,
                  "families": {}, "test_outcomes_used": False}
        for family in FAMILIES:
            options = []
            for policy in POLICIES:
                policy_family, mix, beta = policy_parameters(policy)
                if policy_family != family and not (family == "wise_ft" and policy == "supported"):
                    continue
                for rate in config.learning_rates:
                    for alpha in ALPHAS if family == "wise_ft" else (1.,):
                        parameter = (mix, beta) if family == "allocation_distillation" else mix if family == "allocation" else alpha if family == "wise_ft" else beta
                        seed_histories = {seed: [{**next(v for v in row["validation"] if v["alpha"] == alpha),
                                                 "epoch": row["epoch"], "state_id": row["state_id"]}
                                                for row in histories[(policy, rate, seed)]] for seed in config.seeds}
                        options.append({"policy": policy, "learning_rate": rate, "alpha": alpha,
                                        "parameter": parameter, "histories": seed_histories})
            chosen = choose_family(options, config.seeds, tolerance)
            policy, rate, alpha = chosen["policy"], chosen["learning_rate"], chosen["alpha"]
            runs = []
            for seed, best in chosen["best"].items():
                row = histories[(policy, rate, seed)][best["epoch"]]
                state = _state(repository, output, ledger, policy, rate, seed, row, family=family, alpha=alpha)
                if family == "wise_ft":
                    source_checkpoint = torch.load(repository / state["checkpoint"], map_location="cpu", weights_only=True)
                    scaled = {key: value * alpha for key, value in source_checkpoint["state_dict"].items()}
                    sid = f"{state['state_id']}__wise_alpha_{alpha:.8g}"
                    path = output / "selected" / f"{sid}.pt"
                    checkpoint = {**source_checkpoint, "state_dict": scaled, "state_id": sid,
                        "method": "wise_ft", "family": "wise_ft", "alpha": alpha,
                        "parameters_already_scaled": True, "source_checkpoint_sha256": state["checkpoint_sha256"],
                        "update_norm": update_norm(scaled)}
                    if not path.exists():
                        atomic_torch_save(path, checkpoint)
                    else:
                        previous = torch.load(path, map_location="cpu", weights_only=True)
                        if (previous.get("source_checkpoint_sha256") != state["checkpoint_sha256"] or
                            previous.get("alpha") != alpha or
                            any(not torch.equal(scaled[k], previous["state_dict"][k]) for k in scaled)):
                            raise ValueError("Previously exported WiSE checkpoint changed")
                    state = {**state, "state_id": sid, "method": "wise_ft",
                             "checkpoint": str(path.resolve().relative_to(repository.resolve())),
                             "checkpoint_sha256": sha256_file(path), "parameters_already_scaled": True}
                    if not any(s["state_id"] == sid for s in states):
                        states.append(state)
                runs.append({**state, "validation": {k: best[k] for k in ("native", "source_retrieval")},
                             "nonzero_learned": best["epoch"] > 0 and best["update_norm"] > 0})
                selections.append({"state_id": state["state_id"], "family": family,
                                   "seed": seed, "tolerance_pp": tolerance})
            result["families"][family] = {k: v for k, v in chosen.items() if k != "best"}
            result["families"][family].update({"runs": runs, "all_three_nonzero":
                                             len(runs) == 3 and all(r["nonzero_learned"] for r in runs)})
        path = output / f"selection_{name}.json"
        serialized = json.loads(canonical_json(result))
        if path.exists() and json.loads(path.read_text()) != serialized:
            raise ValueError("An existing extension selection is immutable")
        if not path.exists():
            atomic_json(path, result)
    primary = json.loads((output / "selection_primary.json").read_text())
    joint = primary["families"]["allocation_distillation"]
    _, selected_mix, selected_beta = policy_parameters(joint["policy"])
    decomposition = []
    cells = {"U": "supported", "A": f"allocation_{selected_mix:g}",
             "D": f"distilled_{selected_beta:g}", "AD": joint["policy"]}
    for run in joint["runs"]:
        for cell, policy in cells.items():
            row = histories[(policy, joint["learning_rate"], run["seed"])][run["epoch"]]
            decomposition.append({"cell": {"U": "supported", "A": "allocation", "D": "distilled",
                                          "AD": "allocation_distillation"}[cell],
                "cell_symbol": cell, "state_id": row["state_id"],
                "seed": run["seed"], "epoch": run["epoch"], "learning_rate": joint["learning_rate"],
                "source_mix": selected_mix if cell in ("A", "AD") else 0.,
                "beta": selected_beta if cell in ("D", "AD") else 0.,
                "conditioning": "primary_A+D_development_selected_schedule"})
    manifest = {"schema_version": 1, "extension_id": EXTENSION_ID,
        "ledger_sha256": ledger["ledger_sha256"], "protocol_sha256": protocol_hash,
        "encoder": ledger["identity"]["encoder"], "architecture": "linear",
        "logit_scale": ledger["identity"]["inputs"]["logit_scale"], "states": states,
        "selections": selections, "decomposition_selections": decomposition,
        "test_outcomes_used_for_selection": False, "matched_controls_complete": False,
        "selection_sha256": {name: sha256_file(output / f"selection_{name}.json")
                             for name in ("primary", "sensitivity")}}
    path = output / "state_manifest.json"
    if path.exists() and json.loads(path.read_text()).get("matched_controls_complete"):
        previous = json.loads(path.read_text())
        if previous["selection_sha256"] != manifest["selection_sha256"]:
            raise ValueError("Completed matched controls bind a different selection")
        return previous
    if path.exists() and json.loads(path.read_text()) != json.loads(canonical_json(manifest)):
        raise ValueError("An existing extension state manifest is immutable")
    if not path.exists():
        atomic_json(path, manifest)
    return manifest


def fit_matched_controls(repository: Path, output: Path, config: StudyConfig, data: FeatureDataset,
                         pool: SourceRetrievalPool, ledger: dict, assignments: dict) -> dict:
    """Fit three promotion draws using the selected A+D schedule, without tuning."""
    selection_path = output / "selection_primary.json"
    selected = json.loads(selection_path.read_text())
    manifest = json.loads((output / "state_manifest.json").read_text())
    if selected["ledger_sha256"] != ledger["ledger_sha256"] or manifest["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Matched controls require this ledger's frozen selection")
    selection_hash = sha256_file(selection_path)
    if manifest["selection_sha256"]["primary"] != selection_hash:
        raise ValueError("Selection changed before matched controls")
    chosen = selected["families"]["allocation_distillation"]
    policy, rate = chosen["policy"], chosen["learning_rate"]
    for run in chosen["runs"]:
        seed, epoch = run["seed"], run["epoch"]
        for draw in range(3):
            method = f"matched_allocation_distillation_draw_{draw}"
            fit_candidate(repository, output, config, data, pool, ledger, method, rate, seed,
                          assignment=assignments[draw], target_epochs=epoch, loss_policy=policy,
                          selection_sha256=selection_hash, draw_id=draw)
            directory = candidate_path(output, method, rate, seed)
            row = json.loads((directory / "history.json").read_text())["history"][-1]
            state = _state(repository, output, ledger, method, rate, seed, row,
                           family="matched_allocation_distillation", loss_policy=policy, draw_id=draw)
            state.update({"assignment_seed": ASSIGNMENT_SEEDS[draw], "selection_sha256": selection_hash,
                          "loss_policy": policy})
            if not any(s["state_id"] == state["state_id"] for s in manifest["states"]):
                manifest["states"].append(state)
                manifest["selections"].append({"state_id": state["state_id"],
                    "family": "matched_allocation_distillation", "seed": seed,
                    "draw_id": draw, "tolerance_pp": 1.})
    manifest["matched_controls_complete"] = True
    atomic_json(output / "state_manifest.json", manifest)
    return manifest
