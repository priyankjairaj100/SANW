#!/usr/bin/env python3
"""Bounded, train/development-only text tower feasibility and parity check.

This does not train, score held-out outcomes, or produce a large activation cache.
The manifest is parsed once, but text content is dereferenced only after its
image split is established as train or validation. Calibration and test text
are excluded. A future full cache operation requires a separately locked plan.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gcr.practical_text_v6 import WEIGHTS, digest, load_text_tower, tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=WEIGHTS, required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.examples <= 128:
        parser.error("Bounded feasibility supports at most 128 examples")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    manifest_path = ROOT / "data/visual_entailment/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    allowed_images = {item["id"]: item["split"] for item in manifest["images"]
                      if item["split"] in ("train", "validation")}
    allowed_ids = {pair["text_id"] for pair in manifest["pairs"]
                   if pair["image_id"] in allowed_images}
    train_ids = {pair["text_id"] for pair in manifest["pairs"]
                 if allowed_images.get(pair["image_id"]) == "train"}
    rows = [item for item in manifest["texts"] if item["id"] in allowed_ids]
    encode = tokenizer()
    started = time.monotonic()
    tokens = encode([item["text"] for item in rows])
    lengths = tokens.argmax(dim=-1).numpy() + 1
    tokenize_seconds = time.monotonic() - started
    sample = [i for i, item in enumerate(rows) if item["id"] in train_ids][:args.examples]
    batch = tokens[sample]
    model = load_text_tower(args.encoder, ROOT / "data/practical_v6_models")
    feature_path = (ROOT / "results/resume_features/visual_entailment/features.npz"
                    if args.encoder == "vit_b32" else
                    ROOT / "results/strengthen_second_encoder/features/visual_entailment/features.npz")
    with np.load(feature_path, allow_pickle=False) as archive:
        index = {value: i for i, value in enumerate(archive["text_ids"].tolist())}
        reference = archive["text_features"][[index[rows[i]["id"]] for i in sample]]
    with torch.inference_mode():
        model(batch[:2])
        started = time.monotonic()
        prefix, ends = model.prefix(batch)
        prefix_seconds = time.monotonic() - started
        started = time.monotonic()
        result = model.suffix(prefix, ends)
        suffix_seconds = time.monotonic() - started
        stock_length_result = model(batch, trim=False)
    error = float(np.abs(result.numpy() - reference).max())
    trim_error = float((stock_length_result - result).abs().max())
    if error > 3e-6 or trim_error > 3e-6:
        raise RuntimeError(f"Cached-vector or causal-trimming parity failed: {error}, {trim_error}")
    parameters = model.enable_last_block()
    report = {
        "schema_version": 1,
        "purpose": "train_dev_only_text_adaptation_feasibility",
        "encoder": args.encoder,
        "identity": WEIGHTS[args.encoder],
        "manifest_sha256": digest(manifest_path),
        "reference_features_sha256": digest(feature_path),
        "module_sha256": digest(ROOT / "src/gcr/practical_text_v6.py"),
        "script_sha256": digest(__file__),
        "allowed_splits": ["train", "validation"],
        "calibration_and_test_text_excluded": True,
        "optimizer_updates": 0,
        "large_activation_cache_created": False,
        "text_count": len(rows),
        "train_text_count": len(train_ids),
        "token_lengths": {"mean": float(lengths.mean()), "max": int(lengths.max()),
                          "total": int(lengths.sum())},
        "packed_prefix_float32_bytes": int(lengths.sum()) * 512 * 4,
        "padded_prefix_float32_bytes": len(rows) * 77 * 512 * 4,
        "tokenization_seconds": tokenize_seconds,
        "benchmark_examples": args.examples,
        "benchmark_max_length": int(ends.max()) + 1,
        "benchmark_prefix_seconds": prefix_seconds,
        "benchmark_suffix_forward_seconds": suffix_seconds,
        "reference_parity_max_absolute_error": error,
        "causal_trim_parity_max_absolute_error": trim_error,
        "trainable_last_block_and_ln_parameters": sum(p.numel() for p in parameters),
        "threads": args.threads,
        "torch_version": torch.__version__,
        "runtime_estimates_are_extrapolations": True,
        "prefix_full_split_seconds_linear_extrapolation": prefix_seconds * len(rows) / args.examples,
        "suffix_full_epoch_forward_seconds_linear_extrapolation": suffix_seconds * len(rows) / args.examples,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
