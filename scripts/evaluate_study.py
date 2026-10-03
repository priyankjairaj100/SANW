#!/usr/bin/env python3
"""Evaluate selected adapters and frozen features, saving all item predictions.

Run only after the training selection manifest is frozen. Existing outputs are
resumed only when checkpoint, input and evaluator hashes are identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from gcr.adapters import ResidualAdapter
from gcr.evaluation import (
    PAIR_TIE_POLICY, RETRIEVAL_TIE_POLICY, evaluate_relations,
    evaluate_retrieval, evaluate_triplets,
)

DATASETS = ("visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
EXPECTED = {"visual_entailment": {"images": 400}, "sugarcrepe": {"triplets": 7511, "images": 1560},
            "sugarcrepe_pp": {"triplets": 4757, "images": 1542}, "coco_karpathy": {"images": 5000, "texts": 25000}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_dataset(name: str, allow_prototype: bool) -> dict:
    manifest_path = ROOT / "data" / name / "manifest.json"
    feature_path = ROOT / "results" / "features" / name / "features.npz"
    metadata_path = feature_path.with_name("metadata.json")
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    manifest_hash = sha256(manifest_path)
    recorded_hash = metadata.get("manifest_sha256", metadata.get("input_manifest_sha256"))
    if recorded_hash is not None and recorded_hash != manifest_hash:
        raise ValueError(f"Stale feature cache for {name}")
    with np.load(feature_path, allow_pickle=False) as values:
        features = {key: values[key] for key in ("image_features", "text_features", "image_ids", "text_ids")}
    if features["image_ids"].tolist() != [str(row["id"]) for row in manifest["images"]]:
        raise ValueError(f"Image order differs from manifest for {name}")
    if features["text_ids"].tolist() != [str(row["id"]) for row in manifest["texts"]]:
        raise ValueError(f"Text order differs from manifest for {name}")
    for kind in ("image", "text"):
        values = features[f"{kind}_features"]
        if values.shape != (len(features[f"{kind}_ids"]), 512) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {kind} feature matrix for {name}")
        if not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-4):
            raise ValueError(f"Non-unit {kind} feature rows for {name}")
    mask = np.asarray([row["split"] == "test" for row in manifest["images"]])
    features["image_ids"] = features["image_ids"][mask]
    features["image_features"] = features["image_features"][mask]
    actual_counts = {"images": len(features["image_ids"]), "texts": len(features["text_ids"]),
                     "triplets": len(manifest.get("triplets", []))}
    if not allow_prototype:
        for key, expected in EXPECTED[name].items():
            if actual_counts[key] != expected:
                raise ValueError(f"{name} has {actual_counts[key]} {key}; full study requires {expected}")
    return {"manifest": manifest, "features": features, "metadata": metadata,
            "hashes": {"manifest_sha256": manifest_hash, "features_sha256": sha256(feature_path),
                       "metadata_sha256": sha256(metadata_path)}, "counts": actual_counts}


def adapted_features(features: dict, adapter: ResidualAdapter | None, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    with torch.inference_mode():
        frozen_encoder = lambda values: torch.nn.functional.normalize(values, dim=-1)
        encoders = (("image_features", frozen_encoder if adapter is None else adapter.encode_image),
                    ("text_features", frozen_encoder if adapter is None else adapter.encode_text))
        for key, encoder in encoders:
            values = features[key]
            outputs.append(np.concatenate([
                encoder(torch.from_numpy(np.ascontiguousarray(values[start:start + batch_size], dtype=np.float32))).cpu().numpy()
                for start in range(0, len(values), batch_size)
            ]))
    return outputs[0], outputs[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "results/study/selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--frozen-only", action="store_true")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--allow-prototype", action="store_true", help="Explicitly mark all outputs as pipeline smoke tests")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    selection = {"methods": {}} if args.frozen_only else json.loads(args.selection.read_text())
    selection_hash = None if args.frozen_only else sha256(args.selection)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in
                     (Path(__file__).resolve(), ROOT / "src/gcr/evaluation.py", ROOT / "src/gcr/adapters.py")}
    runs = [{"run_id": "frozen", "method": "frozen", "seed": None, "epoch": 0,
             "checkpoint": None, "checkpoint_sha256": None}]
    for method, entry in sorted(selection["methods"].items()):
        if args.methods is not None and method not in args.methods:
            continue
        for run in sorted(entry["runs"], key=lambda r: r["seed"]):
            if args.seeds is not None and run["seed"] not in args.seeds:
                continue
            checkpoint = Path(run["checkpoint"])
            checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
            actual_hash = sha256(checkpoint)
            if actual_hash != run["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {checkpoint}")
            runs.append({"run_id": f"{method}_seed_{run['seed']}", "method": method, "seed": run["seed"],
                         "epoch": run["epoch"], "learning_rate": entry["learning_rate"],
                         "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": actual_hash})
    datasets = {name: load_dataset(name, args.allow_prototype) for name in args.datasets}
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "torch_threads": torch.get_num_threads(), "blas_thread_environment":
                   {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    index = {"schema_version": 1, "status": "running", "evidence_type": "pipeline_smoke_test" if args.allow_prototype else "new_execution",
             "selection_sha256": selection_hash, "selection_ledger_sha256": selection.get("ledger_sha256"),
             "source_hashes": source_hashes, "environment": environment,
             "retrieval_tie_policy": RETRIEVAL_TIE_POLICY, "pair_tie_policy": PAIR_TIE_POLICY,
             "score_scale": "unscaled float64 dot product; positive fixed CLIP scale does not affect these rank metrics",
             "frozen_normalization": "same final float32 L2 normalization as the zero-initialized adapter",
             "runs": []}
    args.output.mkdir(parents=True, exist_ok=True)
    for run in runs:
        adapter = None
        if run["checkpoint"]:
            payload = torch.load(ROOT / run["checkpoint"], map_location="cpu", weights_only=True)
            if payload["method"] != run["method"] or payload["seed"] != run["seed"] or payload["epoch"] != run["epoch"]:
                raise ValueError(f"Checkpoint metadata mismatch: {run['run_id']}")
            if payload["ledger_sha256"] != selection["ledger_sha256"]:
                raise ValueError(f"Checkpoint protocol ledger mismatch: {run['run_id']}")
            adapter = ResidualAdapter(dim=512)
            adapter.load_state_dict(payload["state_dict"], strict=True)
            adapter.eval()
        output_run = dict(run, datasets={})
        for name, dataset in datasets.items():
            start = time.monotonic()
            output_dir = args.output / run["run_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path, metrics_path = output_dir / f"{name}.npz", output_dir / f"{name}.json"
            provenance = {"run": run, "dataset": name, "input_hashes": dataset["hashes"], "source_hashes": source_hashes,
                          "selection_sha256": selection_hash, "evidence_type": index["evidence_type"],
                          "score_dtype": "float64", "adaptation_dtype": "float32", "environment": environment,
                          "retrieval_tie_policy": RETRIEVAL_TIE_POLICY, "pair_tie_policy": PAIR_TIE_POLICY}
            reused = False
            if metrics_path.exists() and predictions_path.exists() and not args.overwrite:
                saved = json.loads(metrics_path.read_text())
                if saved["provenance"] != provenance or saved["predictions_sha256"] != sha256(predictions_path):
                    raise ValueError(f"Existing output has different provenance: {metrics_path}; use another output or --overwrite")
                metrics = saved["metrics"]
                reused = True
            else:
                image_features, text_features = adapted_features(dataset["features"], adapter)
                shared = (image_features, text_features, dataset["features"]["image_ids"], dataset["features"]["text_ids"])
                manifest = dataset["manifest"]
                if name == "visual_entailment":
                    metrics, predictions = evaluate_relations(*shared, manifest["pairs"])
                    if not args.allow_prototype and metrics["excluded_images"]:
                        raise ValueError("Full e-SNLI test requires all 400 images to have supported/contradicted labels")
                elif name == "coco_karpathy":
                    metrics, predictions = evaluate_retrieval(*shared, manifest["pairs"], block_size=args.block_size)
                else:
                    metrics, predictions = evaluate_triplets(*shared, manifest["triplets"])
                np.savez_compressed(predictions_path, **predictions)
                write_json(metrics_path, {"metrics": metrics, "provenance": provenance,
                                          "predictions_sha256": sha256(predictions_path),
                                          "seconds": time.monotonic() - start})
            output_run["datasets"][name] = {"metrics": metrics, "predictions": str(predictions_path.relative_to(ROOT)),
                                            "metadata": str(metrics_path.relative_to(ROOT)),
                                            "predictions_sha256": sha256(predictions_path)}
            print(json.dumps({"run": run["run_id"], "dataset": name, "reused": reused,
                              "seconds": round(time.monotonic() - start, 2), "metrics": metrics}), flush=True)
        index["runs"].append(output_run)
        write_json(args.output / "index.json", index)
    index["status"] = "complete"
    index["run_count"] = len(index["runs"])
    index["complete_selected_study"] = len(index["runs"]) == 37 and set(args.datasets) == set(DATASETS) and not args.allow_prototype
    write_json(args.output / "index.json", index)


if __name__ == "__main__":
    main()
