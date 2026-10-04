#!/usr/bin/env python3
"""Prepare first-11-block text prefixes for authorized training/development only."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gcr.practical_text_v6 import digest, load_text_tower, tokenizer, WEIGHTS
from gcr.practical_training import input_paths

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", choices=WEIGHTS, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    temporary = output.with_name(output.name + ".building")
    if output.exists() or temporary.exists():
        raise FileExistsError("Choose a new cache path; existing bytes are preserved")
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("Positive thread and batch counts required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    paths = input_paths(ROOT, args.encoder)
    rows, arrays, input_hashes = [], [], {}
    for category, mk, fk in [("visual_entailment", "manifest", "features"),
                              ("retrieval_development", "development_manifest", "development_features")]:
        manifest = json.loads(paths[mk].read_text())
        if category == "visual_entailment":
            allowed_images = {v["id"] for v in manifest["images"] if v["split"] in ("train", "validation")}
            ids = {v["text_id"] for v in manifest["pairs"] if v["image_id"] in allowed_images}
        else:
            # This exact dev900 manifest is the permitted development gallery.
            if len(manifest["images"]) != 900 or len(manifest["texts"]) != 4500:
                raise RuntimeError("Unexpected development gallery")
            ids = {v["id"] for v in manifest["texts"]}
        selected = [v for v in manifest["texts"] if v["id"] in ids]
        with np.load(paths[fk], allow_pickle=False) as archive:
            lookup = {v: i for i, v in enumerate(archive["text_ids"].tolist())}
            values = archive["text_features"][[lookup[v["id"]] for v in selected]]
            arrays.append(F.normalize(torch.from_numpy(values), dim=-1).numpy())
        rows.extend({"id": v["id"], "text": v["text"], "category": category} for v in selected)
        input_hashes[mk] = {"path": str(paths[mk].relative_to(ROOT)), "sha256": digest(paths[mk])}
        input_hashes[fk] = {"path": str(paths[fk].relative_to(ROOT)), "sha256": digest(paths[fk])}
    # Namespaced IDs avoid collisions between separately built manifests.
    keys = [v["category"] + ":" + v["id"] for v in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate namespaced text ID")
    tokens = tokenizer()([v["text"] for v in rows])
    lengths = tokens.argmax(dim=-1).numpy() + 1
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    temporary.mkdir(parents=True)
    np.save(temporary / "keys.npy", np.asarray(keys), allow_pickle=False)
    np.save(temporary / "offsets.npy", offsets, allow_pickle=False)
    np.save(temporary / "tokens.npy", tokens.numpy(), allow_pickle=False)
    teachers = np.concatenate(arrays)
    np.save(temporary / "teacher_features.npy", teachers, allow_pickle=False)
    prefixes = np.lib.format.open_memmap(temporary / "prefix_values.npy", mode="w+", dtype=np.float32,
                                        shape=(int(offsets[-1]), 512))
    reencoded = np.lib.format.open_memmap(temporary / "initial_reencoded_features.npy", mode="w+", dtype=np.float32,
                                         shape=(len(rows), WEIGHTS[args.encoder]["dimension"]))
    tower = load_text_tower(args.encoder, ROOT / "data/practical_v6_models")
    order = np.argsort(lengths, kind="stable")
    maximum_error = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            prefix, ends = tower.prefix(tokens[indices])
            result = tower.suffix(prefix, ends).numpy()
            error = float(np.abs(result - teachers[indices]).max())
            maximum_error = max(maximum_error, error)
            if error > 3e-6:
                raise RuntimeError(f"Frozen pooled-feature parity failed: {error}")
            for row, index in enumerate(indices):
                prefixes[offsets[index]:offsets[index + 1]] = prefix[row, :lengths[index]].numpy()
            reencoded[indices] = result
            if start % (args.batch_size * 25) == 0 or start + args.batch_size >= len(rows):
                print(json.dumps({"encoder": args.encoder, "completed": min(start + args.batch_size, len(rows)),
                                  "total": len(rows), "elapsed_seconds": time.monotonic() - started,
                                  "parity_max_error": maximum_error}), flush=True)
    prefixes.flush(); reencoded.flush()
    del prefixes, reencoded, tower
    report = {"schema_version": 1, "encoder": args.encoder, "weight_identity": WEIGHTS[args.encoder],
              "text_count": len(rows), "train_validation_count": len(arrays[0]),
              "retrieval_development_count": len(arrays[1]), "token_count": int(offsets[-1]),
              "prefix_dtype": "float32", "prefix_shape": [int(offsets[-1]), 512],
              "train_validation_only": True, "test_and_calibration_excluded": True,
              "optimizer_steps": 0, "maximum_cached_feature_parity_error": maximum_error,
              "inputs": input_hashes, "threads": args.threads, "batch_size": args.batch_size,
              "elapsed_seconds": time.monotonic() - started,
              "source_sha256": {"src/gcr/practical_text_v6.py": digest(ROOT / "src/gcr/practical_text_v6.py"),
                                 "scripts/cache_practical_text_prefix_v6.py": digest(__file__)},
              "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                        for path in sorted(temporary.glob("*.npy"))}}
    (temporary / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.rename(output)
    print(json.dumps({"complete": str(output), "receipt_sha256": digest(output / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
