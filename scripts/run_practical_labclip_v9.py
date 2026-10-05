#!/usr/bin/env python3
"""Fit protocol-bound LABCLIP controls using original training owners only.

No evaluation data are loaded. An independent v9 evaluator owns internal
selection, official development evaluation, and any eventual practical gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcr.practical_labclip_v9 import (
    LABCLIPConfig, OFFICIAL_FILES, STUDY, canonical_transform,
    fit_alignment, sources_by_owner, validate_owner_split,
)
from run_practical_constrained_v8 import load_training, paths_for


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def serial_config(config):
    result = asdict(config)
    result["checkpoint_epochs"] = list(result["checkpoint_epochs"])
    return result


def fitting_source_paths():
    return ("src/gcr/practical_labclip_v9.py", "scripts/run_practical_labclip_v9.py",
            "scripts/run_practical_constrained_v8.py", "src/gcr/training.py",
            "src/gcr/practical_constrained_v8.py")


def verify_protocol(repository, protocol_path, encoder, arm, seed, mode, selection_path=None,
                    replication_gate_path=None):
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("study") != "sanw_practical_v9":
        raise ValueError("Wrong immutable v9 protocol study")
    if encoder not in protocol.get("encoders", []) or seed not in protocol.get("seeds", []):
        raise ValueError("Encoder or seed is absent from the immutable protocol")
    config = LABCLIPConfig.published_control(seed) if arm == "published_control" else LABCLIPConfig(seed=seed)
    config.validate()
    expected = serial_config(config); expected.pop("seed")
    declared = protocol.get("labclip", {}).get("arms", {}).get(arm, {}).get("config")
    if declared != expected:
        raise ValueError("LABCLIP configuration differs from the immutable protocol")
    if protocol.get("labclip", {}).get("epochs") != [1, 2, 4, 8, 16, 32]:
        raise ValueError("The finite adapted epoch grid changed")
    sources = {p: digest(repository / p) for p in fitting_source_paths()}
    if any(protocol.get("source_sha256", {}).get(p) != sha for p, sha in sources.items()):
        raise ValueError("Fitting source differs from the immutable protocol")
    actual_inputs = {name: {"path": str(path.relative_to(repository)), "sha256": digest(path)}
                     for name, path in paths_for(repository, encoder).items()}
    if protocol.get("training_inputs", {}).get(encoder) != actual_inputs:
        raise ValueError("Training inputs differ from the immutable protocol")
    split_record = protocol.get("split", {})
    split_path = repository / split_record.get("path", "")
    if not split_path.is_file() or digest(split_path) != split_record.get("sha256"):
        raise ValueError("Owner split differs from the immutable protocol")
    split = json.loads(split_path.read_text())
    if split.get("manifest") != actual_inputs["manifest"]:
        raise ValueError("Owner split is bound to a different manifest")
    protocol_sha = digest(protocol_path)
    stop_epoch = config.epochs
    selection_record = None
    if mode == "inner":
        if seed != 17 or selection_path is not None or replication_gate_path is not None:
            raise ValueError("Internal selection fits only seed17, without a selection override")
    elif mode == "full":
        if selection_path is None:
            raise ValueError("Full training requires the immutable passing cross-encoder inner selection")
        selection = json.loads(selection_path.read_text())
        stop_epoch = selection.get("selected_epoch")
        if (selection.get("study") != "sanw_practical_v9_selection" or selection.get("passed") is not True
                or selection.get("family") != "labclip" or selection.get("arm") != arm
                or arm != "small_data_fixed_scale" or selection.get("protocol_sha256") != protocol_sha
                or selection.get("split") != split_record or stop_epoch not in config.checkpoint_epochs):
            raise ValueError("Full fit is not authorized by the shared passing inner selection")
        selection_record = {"path": str(selection_path.resolve()), "sha256": digest(selection_path)}
        if seed != 17:
            if replication_gate_path is None:
                raise ValueError("Seed29/43 full fits require a passing fixed official development gate")
            gate = json.loads(replication_gate_path.read_text())
            if (gate.get("study") != "sanw_practical_v9_replication_gate" or gate.get("passed") is not True
                    or gate.get("family") != "labclip" or gate.get("protocol_sha256") != protocol_sha
                    or gate.get("selection_sha256") != digest(selection_path)
                    or gate.get("encoders_passed") != ["vit_b32", "rn50"]):
                raise ValueError("Replication gate does not authorize this selected family")
            selection_record["replication_gate"] = {"path": str(replication_gate_path.resolve()),
                                                    "sha256": digest(replication_gate_path)}
    else:
        raise ValueError("Unknown fitting mode")
    return config, protocol, split, sources, actual_inputs, stop_epoch, selection_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--arm", choices=("published_control", "small_data_fixed_scale"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mode", choices=("inner", "full"), default="inner")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--replication-gate", type=Path)
    args = parser.parse_args()
    repository, output, protocol_path = args.repository.resolve(), args.output.resolve(), args.protocol.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing comparator fit")
    config, protocol, split, sources, inputs, stop_epoch, selection = verify_protocol(
        repository, protocol_path, args.encoder, args.arm, args.seed, args.mode, args.selection, args.replication_gate)
    import torch
    torch.set_num_threads(config.threads)
    start = time.monotonic()
    images, texts, source_rows, owner, positive, negative, provenance = load_training(repository, args.encoder, "both")
    inner_train, inner_validation = validate_owner_split(split, provenance["training_image_manifest_indices"])
    if len(inner_train) != 960 or len(inner_validation) != 240 or len(images) != 1200:
        raise ValueError("Expected the declared 960/240 split of 1200 original training owners")
    requested_fit_owners = inner_train if args.mode == "inner" else np.arange(len(images), dtype=np.int64)
    # Both controls need one annotated contradiction per source-caption row.
    # Do not invent a negative for owners lacking a valid contradiction.
    fit_owners = np.asarray([i for i in requested_fit_owners if len(negative[i])], dtype=np.int64)
    omitted_owners = np.asarray([i for i in requested_fit_owners if not len(negative[i])], dtype=np.int64)
    if not len(fit_owners):
        raise ValueError("No fitting owners have valid annotated contradictions")
    if any(len(rows) != 5 for rows in sources_by_owner(source_rows, owner, len(images))):
        raise ValueError("The declared data must contain exactly five owned source captions per image")
    metadata = json.loads((repository / inputs["metadata"]["path"]).read_text())
    native_scale = float(metadata["logit_scale"])
    identity = {"study": STUDY, "mode": args.mode, "encoder": args.encoder, "config": serial_config(config),
                "stop_epoch": stop_epoch, "protocol": {"path": str(protocol_path), "sha256": digest(protocol_path)},
                "split": protocol["split"], "selection": selection, "training_provenance": provenance,
                "requested_fit_loader_owner_indices": requested_fit_owners.tolist(),
                "fit_loader_owner_indices": fit_owners.tolist(), "source_sha256": sources,
                "eligible_owner_count": len(fit_owners), "requested_owner_count": len(requested_fit_owners),
                "omitted_no_valid_contradiction_loader_owner_indices": omitted_owners.tolist(),
                "source_caption_rows_per_epoch": 5 * len(fit_owners),
                "updates_per_epoch": ((len(fit_owners) * 5 + config.batch_size - 1) // config.batch_size
                                      if args.arm == "published_control" else
                                      5 * ((len(fit_owners) + config.batch_size - 1) // config.batch_size)),
                "supervision_difference_from_joint": "both_LABCLIP_arms_omit_no_contradiction_owners; joint_retrieval_uses_all_requested_owners",
                "native_logit_scale": native_scale, "official_repository": "kdariina/CLIP-not-BoW-unimodally",
                "official_file_git_blob_sha": OFFICIAL_FILES,
                "score": "pure_image_dot_normalize_full_rank_transformed_text_no_reference_correction",
                "canonical_inference": "float64_rowwise_einsum_then_fixed_norm_then_fixed_pair_sum",
                "frozen_comparator": "unchanged_original_float32_normalized_cached_feature_cosine",
                "identity_parity": "must_be_measured_separately_by_independent_inner_evaluator",
                "data_departure_from_published": "existing_annotated_owner_contradictions_instead_of_synthetic_noun_adjective_shuffles",
                "positive_role": "source_caption_only; supported_texts_not_used_as_loss_targets",
                "supported_label_use": "negative_conflict_exclusion_only",
                "control": "published_control_keeps_diagonal_targets_and_repeated_image_rows",
                "adapted_departures": "unique_image_batches_five_source_passes;native_fixed_scale;B128;32epochs",
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_sha})
    checkpoint_rows = []
    train_sources = source_rows[np.isin(owner, fit_owners)]
    reference_direction = canonical_transform(texts[train_sources], np.eye(images.shape[1]))

    def on_checkpoint(row, model):
        checkpoint = output / "checkpoints" / f"epoch_{row['epoch']:03d}.npz"
        model.save(checkpoint)
        weight = model.linear.weight.detach().cpu().numpy().astype(np.float64)
        scalar = float(np.trace(weight) / len(weight))
        direction = canonical_transform(texts[train_sources], weight)
        direction_change = direction - reference_direction
        non_scalar = float(np.linalg.norm(weight - scalar * np.eye(len(weight))))
        # This scan is training-only and tests functional movement, not success.
        max_score_change = 0.0
        for lo in range(0, len(fit_owners), 64):
            max_score_change = max(max_score_change, float(np.abs(images[fit_owners[lo:lo + 64]] @ direction_change.T).max()))
        item = {**row, "checkpoint": {"path": str(checkpoint.relative_to(output)), "sha256": digest(checkpoint)},
                "non_scalar_weight_norm": non_scalar,
                "weight_identity_distance": float(np.linalg.norm(weight - np.eye(len(weight)))),
                "max_training_source_direction_change": float(np.linalg.norm(direction_change, axis=1).max()),
                "max_training_source_score_change_from_identity": max_score_change,
                "nonzero_functional_update": bool(non_scalar > 1e-12 and max_score_change > 1e-12),
                "ledger_sha256": ledger_sha}
        checkpoint_rows.append(item)
        atomic_json(output / "checkpoint_history.json", checkpoint_rows)
        print(json.dumps({"encoder": args.encoder, "arm": args.arm, "epoch": row["epoch"],
                          "loss": row["loss"], "logit_scale": row["logit_scale"],
                          "nonzero_functional_update": item["nonzero_functional_update"]}), flush=True)

    model, result = fit_alignment(images, texts, source_rows, owner, negative, fit_owners,
                                  config, native_scale, on_checkpoint, stop_epoch=stop_epoch)
    atomic_json(output / "history.json", result["history"])
    result.update({"study": STUDY, "mode": args.mode, "encoder": args.encoder, "arm": args.arm,
                   "ledger_sha256": ledger_sha, "checkpoint_history": checkpoint_rows,
                   "train_only": True, "official_development_or_test_read": False,
                   "eligible_owner_count": len(fit_owners), "requested_owner_count": len(requested_fit_owners),
                   "omitted_no_valid_contradiction_loader_owner_indices": omitted_owners.tolist(),
                   "validation_owners_used_for_optimization": args.mode == "full",
                   "elapsed_seconds": time.monotonic() - start})
    atomic_json(output / "completion.json", result)
    print(json.dumps({"complete": True, "encoder": args.encoder, "arm": args.arm,
                      "mode": args.mode, "optimizer_steps": result["optimizer_steps"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
