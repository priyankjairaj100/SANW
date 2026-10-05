#!/usr/bin/env python3
"""Train-only preflight or fitting for the v8 constrained residual.

This runner never loads development galleries or benchmark test datasets.
The shared original feature archive is sliced to training rows before all PCA,
centering, labels, mining, loss calculations, and exported predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import unicodedata
import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcr.practical_constrained_v8 import (
    CompositionExamples, ConstrainedBilinearScorer, FitConfig,
    FullGalleryConstraints, fit, radius_headroom,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def paths_for(repository, encoder):
    parent = (repository / "results/resume_features/visual_entailment" if encoder == "vit_b32"
              else repository / "results/strengthen_second_encoder/features/visual_entailment")
    return {"manifest": repository / "data/visual_entailment/manifest.json",
            "features": parent / "features.npz", "metadata": parent / "metadata.json"}


def caption_key(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def load_training(repository, encoder, relation_type):
    # The existing validated loader and exact Torch normalization retain baseline
    # parity with prior experiments; no Torch is required by the new model itself.
    import torch
    import torch.nn.functional as F
    from gcr.training import FeatureDataset

    paths = paths_for(repository, encoder)
    data = FeatureDataset(repository, paths["manifest"], paths["features"], paths["metadata"],
                          512 if encoder == "vit_b32" else 1024)
    image_rows = list(data.split_indices["train"])
    text_rows = sorted({j for i in image_rows for j in data.pairs[i]})
    lookup = {j: k for k, j in enumerate(text_rows)}
    images = F.normalize(data.images[image_rows], dim=-1).numpy().astype(np.float64)
    texts = F.normalize(data.texts[text_rows], dim=-1).numpy().astype(np.float64)
    source_global = sorted({j for i in image_rows for j, code in data.pairs[i].items() if code == 1})
    source_lookup = {j: k for k, j in enumerate(source_global)}
    owner = np.full(len(source_global), -1, dtype=np.int64)
    positive, negative, excluded = [], [], []
    relation_codes = {"supported": (2,), "source": (1,), "both": (1, 2)}[relation_type]
    for local_i, global_i in enumerate(image_rows):
        relations = data.pairs[global_i]
        positive_keys = {caption_key(data.manifest["texts"][j]["text"])
                         for j, code in relations.items() if code in (1, 2)}
        positive.append(sorted(lookup[j] for j, code in relations.items() if code in relation_codes))
        good_negative = []
        for j, code in relations.items():
            if code == 1:
                column = source_lookup[j]
                if owner[column] >= 0:
                    raise ValueError("Training source caption has multiple owners")
                owner[column] = local_i
            elif code == 3:
                if caption_key(data.manifest["texts"][j]["text"]) in positive_keys:
                    excluded.append({"training_image_index": local_i, "global_text_index": j})
                else:
                    good_negative.append(lookup[j])
        negative.append(sorted(good_negative))
    source_rows = np.asarray([lookup[j] for j in source_global], dtype=np.int64)
    provenance = {"inputs": {k: {"path": str(v.relative_to(repository)), "sha256": digest(v)} for k, v in paths.items()},
                  "training_image_manifest_indices": image_rows, "training_text_manifest_indices": text_rows,
                  "training_source_text_manifest_indices": source_global,
                  "excluded_exact_normalized_contradiction_conflicts": excluded,
                  "basis_text_scope": "all distinct training-associated captions including neutral rows as unlabelled PCA covariates",
                  "feature_preprocessing": "existing_loader_then_Torch_float32_F_normalize_once_then_float64",
                  "torch_version": torch.__version__, "fit_split": "train", "heldout_used": False,
                  "caption_equivalence_assumed": False, "neutral_as_negative": False}
    return images, texts, source_rows, owner, positive, negative, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnose", "fit"))
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--diagnostic-radii", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    for field in fields(FitConfig):
        default = getattr(FitConfig(), field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    config = FitConfig(**{field.name: getattr(args, field.name) for field in fields(FitConfig)})
    config.validate()
    repository, output = args.repository.resolve(), args.output.resolve()
    source_paths = [repository / "src/gcr/practical_constrained_v8.py", Path(__file__).resolve(),
                    repository / "src/gcr/training.py", repository / "src/gcr/practical_constrained_evaluation_v8.py"]
    source_hashes = {str(p.relative_to(repository)): digest(p) for p in source_paths}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run")
    protocol = None
    if args.mode == "fit":
        if args.protocol is None:
            raise ValueError("Fit requires an immutable protocol recorded before fitting")
        protocol = json.loads(args.protocol.read_text())
        expected = asdict(config); expected.pop("seed")
        if protocol.get("study") != "sanw_constrained_bilinear_v8" or protocol.get("fit_config") != expected:
            raise ValueError("Fit configuration does not match the immutable protocol")
        if config.seed not in protocol.get("seeds", []):
            raise ValueError("Seed is absent from the protocol")
        if args.encoder not in protocol.get("encoders", []):
            raise ValueError("Encoder is absent from the protocol")
        declared_sources = protocol.get("source_sha256", {})
        if any(declared_sources.get(path) != sha for path, sha in source_hashes.items()):
            raise ValueError("Source code differs from the immutable protocol")
        actual_inputs = {name: {"path": str(path.relative_to(repository)), "sha256": digest(path)}
                         for name, path in paths_for(repository, args.encoder).items()}
        if protocol.get("training_inputs", {}).get(args.encoder) != actual_inputs:
            raise ValueError("Training input paths or hashes differ from the immutable protocol")
    start = time.monotonic()
    images, texts, source_rows, owner, positive, negative, provenance = load_training(repository, args.encoder, config.relation_type)
    scorer = ConstrainedBilinearScorer.from_training(images, texts, config.rank)
    composition = CompositionExamples(images, texts, positive, negative, scorer)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"study": "sanw_constrained_bilinear_v8", "encoder": args.encoder, "mode": args.mode,
                "config": asdict(config), "training_provenance": provenance,
                "source_sha256": source_hashes,
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "score": "frozen_cosine_plus_double_centered_bilinear_no_task_or_gallery_switch",
                "train_only": True}
    if args.protocol is not None:
        identity["protocol"] = {"path": str(args.protocol.resolve()), "sha256": digest(args.protocol)}
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ledger_hash = hashlib.sha256(identity_bytes).hexdigest()
    atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_hash})
    if args.mode == "diagnose":
        result = radius_headroom(scorer, composition, args.diagnostic_radii)
        result["encoder"] = args.encoder
        result["ledger_sha256"] = ledger_hash
        result["elapsed_seconds"] = time.monotonic() - start
        atomic_json(output / "training_headroom.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    constraints = FullGalleryConstraints(images, texts[source_rows], owner, scorer, config.retention_fraction)
    checkpoint_rows = []
    (output / "checkpoints").mkdir()

    def on_epoch(row, coefficient):
        scorer.coefficient = coefficient
        checkpoint = output / "checkpoints" / f"epoch_{row['epoch']:03d}.npz"
        scorer.save(checkpoint)
        item = dict(row)
        item["checkpoint"] = {"path": str(checkpoint.relative_to(output)), "sha256": digest(checkpoint), "ledger_sha256": ledger_hash}
        checkpoint_rows.append(item)
        atomic_json(output / "history.json", checkpoint_rows)
        print(json.dumps({"epoch": row["epoch"], "training_objective": row["training_objective"],
                          "pair_accuracy": row["composition"]["image_mean_pair_accuracy"],
                          "active_constraints": row["certificate"]["active_constraints"],
                          "radial_factor": row["certificate"]["radial_restoration_factor"],
                          "coefficient_norm": row["certificate"]["coefficient_frobenius_norm"]}), flush=True)

    result = fit(scorer, composition, constraints, config, on_epoch)
    scorer.save(output / "selected.npz")
    result.update({"encoder": args.encoder, "ledger_sha256": ledger_hash,
                   "selected_checkpoint": {"path": "selected.npz", "sha256": digest(output / "selected.npz")},
                   "checkpoint_history": checkpoint_rows, "elapsed_seconds": time.monotonic() - start})
    atomic_json(output / "completion.json", result)
    print(json.dumps({"complete": True, "selected_epoch": result["selected_epoch"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
