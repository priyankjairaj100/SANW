#!/usr/bin/env python3
"""Fit one v9 joint candidate using its locked inner or full training split.

No official development or benchmark test data are loaded. Inner evaluation and
cross-encoder configuration selection are separate, independently audited CLIs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_practical_constrained_v8 as common
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer, FullGalleryConstraints
from gcr.practical_joint_v9 import JointCompositionExamples, JointFitConfig, FullGallerySourceLoss, fit_joint


def training_view(repository, encoder, selected_manifest_images=None):
    """Select owner-disjoint rows before PCA, means, loss, and constraints.

    All associated caption rows, including unlabelled neutral PCA covariates,
    follow their owner into the selected split. No other captions enter fitting.
    """
    images, texts, source_rows, owner, positives, negatives, provenance = common.load_training(repository, encoder, "both")
    manifest = json.loads((repository / "data/visual_entailment/manifest.json").read_text())
    global_images = provenance["training_image_manifest_indices"]
    global_texts = provenance["training_text_manifest_indices"]
    image_lookup = {manifest["images"][index]["id"]: i for i, index in enumerate(global_images)}
    text_lookup = {manifest["texts"][index]["id"]: i for i, index in enumerate(global_texts)}
    membership = [set() for _ in global_images]
    for pair in manifest["pairs"]:
        if pair["image_id"] in image_lookup:
            membership[image_lookup[pair["image_id"]]].add(text_lookup[pair["text_id"]])
    chosen = global_images if selected_manifest_images is None else list(selected_manifest_images)
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(global_images):
        raise ValueError("Requested fit images must be distinct original training images")
    chosen = sorted(chosen)
    reverse = {index: i for i, index in enumerate(global_images)}
    rows = [reverse[i] for i in chosen]
    row_lookup = {old: new for new, old in enumerate(rows)}
    text_rows = sorted({j for i in rows for j in membership[i]})
    text_remap = {old: new for new, old in enumerate(text_rows)}
    owner_sources = {i: set(source_rows[owner == i].tolist()) for i in rows}
    source_columns = [j for j, old_owner in enumerate(owner) if int(old_owner) in row_lookup]
    new_source_rows = np.asarray([text_remap[int(source_rows[j])] for j in source_columns], dtype=np.int64)
    new_owner = np.asarray([row_lookup[int(owner[j])] for j in source_columns], dtype=np.int64)
    sources = [[text_remap[j] for j in positives[i] if j in owner_sources[i]] for i in rows]
    supported = [[text_remap[j] for j in positives[i] if j not in owner_sources[i]] for i in rows]
    contra = [[text_remap[j] for j in negatives[i]] for i in rows]
    selected_provenance = dict(provenance)
    selected_provenance.update({"training_image_manifest_indices": chosen,
                               "training_text_manifest_indices": [global_texts[j] for j in text_rows],
                               "training_source_text_manifest_indices": [global_texts[int(source_rows[j])] for j in source_columns],
                               "excluded_exact_normalized_contradiction_conflicts": [
                                   {**item, "training_image_index": row_lookup[item["training_image_index"]]}
                                   for item in provenance["excluded_exact_normalized_contradiction_conflicts"]
                                   if item["training_image_index"] in row_lookup]})
    return images[rows], texts[text_rows], new_source_rows, new_owner, sources, supported, contra, selected_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inner", "full"))
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--replication-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--composition-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    repository, output = args.repository.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing candidate")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("study") != "sanw_practical_v9" or args.encoder not in protocol.get("encoders", []):
        raise ValueError("Wrong protocol study or encoder")
    joint = protocol.get("joint", {})
    if args.radius not in joint.get("radii", []) or args.composition_weight not in joint.get("composition_weights", []):
        raise ValueError("Configuration is outside the locked finite grid")
    if args.seed not in joint.get("seeds", []):
        raise ValueError("Seed absent from protocol")
    if args.mode == "inner" and args.seed != joint.get("inner_seed", 17):
        raise ValueError("Inner selection uses only its prespecified seed")
    config = JointFitConfig(**joint["base_config"], radius=args.radius, composition_weight=args.composition_weight, seed=args.seed)
    config.validate()
    source_names = ("src/gcr/practical_joint_v9.py", "scripts/run_practical_joint_v9.py",
                    "src/gcr/practical_constrained_v8.py", "scripts/run_practical_constrained_v8.py",
                    "src/gcr/practical_constrained_evaluation_v8.py", "src/gcr/training.py")
    source_hashes = {name: common.digest(repository / name) for name in source_names}
    if any(protocol.get("source_sha256", {}).get(name) != sha for name, sha in source_hashes.items()):
        raise ValueError("Code differs from protocol")
    actual_inputs = {name: {"path": str(path.relative_to(repository)), "sha256": common.digest(path)}
                     for name, path in common.paths_for(repository, args.encoder).items()}
    if protocol.get("training_inputs", {}).get(args.encoder) != actual_inputs:
        raise ValueError("Training inputs differ from protocol")
    split_info = protocol["split"]
    split_path = repository / split_info["path"]
    if common.digest(split_path) != split_info["sha256"]:
        raise ValueError("Inner split hash differs from protocol")
    split = json.loads(split_path.read_text())
    if split.get("schema") != "sanw_inner_training_owner_split_v9" or split.get("manifest") != actual_inputs["manifest"]:
        raise ValueError("Split schema or input manifest differs")
    train_ids, validation_ids = split["train_image_manifest_indices"], split["validation_image_manifest_indices"]
    if set(train_ids) & set(validation_ids) or len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Invalid or overlapping inner split")
    original_manifest = json.loads((repository / actual_inputs["manifest"]["path"]).read_text())
    original_training = {i for i, image in enumerate(original_manifest["images"]) if image["split"] == "train"}
    if (len(train_ids) != 960 or len(validation_ids) != 240 or len(original_training) != 1200
            or set(train_ids) | set(validation_ids) != original_training):
        raise ValueError("Inner split must partition exactly the original 1200 training owners into 960/240")
    protocol_sha = common.digest(args.protocol)
    selection_info = None
    replication_info = None
    if args.mode == "full":
        if args.selection is None:
            raise ValueError("Full fitting requires a passing shared inner selection")
        selection = json.loads(args.selection.read_text())
        if (selection.get("study") != "sanw_practical_v9_selection" or selection.get("family") != "joint"
                or selection.get("passed") is not True or selection.get("protocol_sha256") != protocol_sha
                or selection.get("split") != split_info
                or selection.get("selected_config") != {"radius": args.radius, "composition_weight": args.composition_weight}):
            raise ValueError("Invalid shared inner selection")
        selection_info = {"path": str(args.selection.resolve()), "sha256": common.digest(args.selection)}
        if args.seed != 17:
            if args.replication_gate is None:
                raise ValueError("Replication seeds require both seed-17 official-development pilots to pass")
            gate = json.loads(args.replication_gate.read_text())
            if (gate.get("study") != "sanw_practical_v9_replication_gate" or gate.get("family") != "joint"
                    or gate.get("passed") is not True or gate.get("protocol_sha256") != protocol_sha
                    or gate.get("selection_sha256") != selection_info["sha256"]
                    or sorted(gate.get("encoders_passed", [])) != ["rn50", "vit_b32"]):
                raise ValueError("Invalid cross-encoder seed-17 replication gate")
            replication_info = {"path": str(args.replication_gate.resolve()), "sha256": common.digest(args.replication_gate)}
    start = time.monotonic()
    selected_ids = train_ids if args.mode == "inner" else sorted(train_ids + validation_ids)
    images, texts, source_rows, owner, sources, supported, contra, provenance = training_view(repository, args.encoder, selected_ids)
    if provenance["training_image_manifest_indices"] != sorted(selected_ids):
        raise ValueError("Resolved training rows differ from locked split")
    native_scale = float(json.loads((repository / actual_inputs["metadata"]["path"]).read_text())["logit_scale"])
    scorer = ConstrainedBilinearScorer.from_training(images, texts, config.rank)
    constraints = FullGalleryConstraints(images, texts[source_rows], owner, scorer, config.retention_fraction)
    composition = JointCompositionExamples(images, texts, sources, supported, contra, scorer)
    retrieval = FullGallerySourceLoss(constraints, native_scale)
    identity = {"study": "sanw_practical_v9", "family": "joint", "mode": args.mode, "encoder": args.encoder,
                "config": asdict(config), "protocol_sha256": protocol_sha,
                "protocol": {"path": str(args.protocol.resolve()), "sha256": protocol_sha}, "split": split_info,
                "selection": selection_info, "replication_gate": replication_info,
                "source_sha256": source_hashes, "training_provenance": provenance,
                "retrieval_logit_scale": native_scale, "fit_gallery_image_count": len(images), "fit_gallery_text_count": len(source_rows),
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "normalization": "same_v8_canonical_frozen_features_and_pair_score",
                "official_development_or_benchmarks_used": False}
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    common.atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_sha})
    checkpoint_rows = []

    def on_epoch(row, coefficient):
        scorer.coefficient = coefficient
        path = output / "checkpoints" / f"epoch_{row['epoch']:03d}.npz"
        scorer.save(path)
        item = dict(row)
        item["checkpoint"] = {"path": str(path.relative_to(output)), "sha256": common.digest(path), "ledger_sha256": ledger_sha}
        checkpoint_rows.append(item)
        common.atomic_json(output / "history.json", checkpoint_rows)
        print(json.dumps({"epoch": row["epoch"], "objective": row["training_objective"], "retrieval_loss": row["retrieval_loss"],
                          "joint_accuracy": row["composition"]["image_mean_joint_accuracy"],
                          "coefficient_norm": row["certificate"]["coefficient_frobenius_norm"],
                          "objective_gap_upper": row["ball_relaxed_convex_suboptimality_upper_bound"]}), flush=True)

    result = fit_joint(scorer, composition, retrieval, constraints, config, on_epoch)
    scorer.save(output / "selected.npz")
    result.update({"study": "sanw_practical_v9", "family": "joint", "mode": args.mode, "encoder": args.encoder,
                   "ledger_sha256": ledger_sha, "protocol_sha256": protocol_sha, "split": split_info,
                   "selected_checkpoint": {"path": "selected.npz", "sha256": common.digest(output / "selected.npz")},
                   "checkpoint_history": checkpoint_rows, "elapsed_seconds": time.monotonic() - start})
    common.atomic_json(output / "completion.json", result)
    print(json.dumps({"complete": True, "selected_epoch": result["selected_epoch"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
