#!/usr/bin/env python3
"""Score only preselected states from the separately frozen exploratory AD study."""
from __future__ import annotations
import argparse
from collections import Counter
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
from gcr.allocation_distillation_analysis import (DATASETS, ENCODERS, FAMILIES, EVIDENCE_TYPE,
    SEEDS, validate_evaluation_specification)
from gcr.evaluation import PAIR_TIE_POLICY, RETRIEVAL_TIE_POLICY, evaluate_relations, evaluate_triplets
from evaluate_study import adapted_features, sha256, write_json
from evaluate_review_followup import compact_retrieval

MANIFEST_PATHS = {name: f"data/{name}/manifest.json" for name in DATASETS}
MANIFEST_PATHS["e_vil_test1000"] = "data/review_followup/e_vil_test1000/manifest.json"
EXPECTED_COUNTS = {"e_vil_test1000": {"images": 1000, "texts": 5000},
    "visual_entailment": {"images": 400}, "sugarcrepe": {"images": 1560, "triplets": 7511},
    "sugarcrepe_pp": {"images": 1542, "triplets": 4757}, "coco_karpathy": {"images": 5000, "texts": 25000}}
SOURCE_PATHS = ("scripts/evaluate_allocation_distillation.py", "scripts/evaluate_study.py",
    "scripts/evaluate_review_followup.py", "src/gcr/adapters.py", "src/gcr/evaluation.py",
    "src/gcr/allocation_distillation_analysis.py")


def path_at_root(repository, value):
    path = Path(value)
    path = (path if path.is_absolute() else repository / path).resolve()
    if not path.is_relative_to(repository.resolve()):
        raise ValueError(f"Artifact path escapes repository: {path}")
    return path


def load_dataset_config(repository, config_path, encoder, ledger):
    """Read exact original manifests with explicit recovered feature-cache paths."""
    config_path = path_at_root(repository, config_path)
    config = json.loads(config_path.read_text())
    if encoder not in ENCODERS or config.get("encoder") != encoder or set(config.get("datasets", {})) != set(DATASETS):
        raise ValueError("Dataset config must identify the encoder and all five benchmark datasets")
    dim = 512 if encoder == "vit_b32" else 1024
    training_metadata = ledger["identity"]["inputs"]["feature_metadata"]
    result = {}
    for name in DATASETS:
        record = config["datasets"][name]
        if set(record) != {"manifest", "features", "metadata"}:
            raise ValueError("Each dataset record requires only manifest, features, and metadata paths")
        paths = {key: path_at_root(repository, value) for key, value in record.items()}
        if paths["manifest"] != (repository / MANIFEST_PATHS[name]).resolve():
            raise ValueError("This extension must use the unchanged original benchmark manifests")
        manifest = json.loads(paths["manifest"].read_text())
        metadata = json.loads(paths["metadata"].read_text())
        hashes = {f"{key}_sha256": sha256(path) for key, path in paths.items()}
        manifest_hash = metadata.get("manifest_sha256", metadata.get("input_manifest_sha256"))
        if manifest_hash != hashes["manifest_sha256"]:
            raise ValueError(f"Feature metadata does not bind its input manifest: {name}")
        if metadata.get("features_sha256") not in (None, hashes["features_sha256"]):
            raise ValueError(f"Feature archive content hash mismatch: {name}")
        for key in ("model_revision", "weights_sha256", "open_clip_version", "logit_scale"):
            if key not in training_metadata or metadata.get(key) != training_metadata[key]:
                raise ValueError(f"Test and training encoder provenance differ: {name} {key}")
        with np.load(paths["features"], allow_pickle=False) as raw:
            features = {key: raw[key] for key in ("image_features", "text_features", "image_ids", "text_ids")}
        for kind in ("image", "text"):
            expected_ids = [str(row["id"]) for row in manifest[f"{kind}s"]]
            ids, values = features[f"{kind}_ids"], features[f"{kind}_features"]
            if ids.tolist() != expected_ids or len(set(expected_ids)) != len(expected_ids):
                raise ValueError(f"Feature order or unique IDs differ from manifest: {name} {kind}")
            if values.shape != (len(ids), dim) or not np.isfinite(values).all():
                raise ValueError(f"Malformed feature matrix: {name} {kind}")
            if not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=1e-4):
                raise ValueError(f"Features must be unit normalized: {name} {kind}")
        mask = np.asarray([row["split"] == "test" for row in manifest["images"]])
        features["image_ids"] = features["image_ids"][mask]
        features["image_features"] = features["image_features"][mask]
        counts = {"images": len(features["image_ids"]), "texts": len(features["text_ids"]),
                  "triplets": len(manifest.get("triplets", []))}
        if any(counts[key] != value for key, value in EXPECTED_COUNTS[name].items()):
            raise ValueError(f"Incomplete benchmark dataset: {name} {counts}")
        if name in ("e_vil_test1000", "coco_karpathy"):
            pairs = manifest["pairs"]
            if len(pairs) != counts["texts"] or any(row["relation"] != "source" for row in pairs):
                raise ValueError("Retrieval requires the complete source-caption ownership labels")
            if Counter(str(row["image_id"]) for row in pairs) != Counter({str(i): 5 for i in features["image_ids"]}):
                raise ValueError("Every retrieval image must own exactly five source captions")
            if Counter(str(row["text_id"]) for row in pairs) != Counter({str(i): 1 for i in features["text_ids"]}):
                raise ValueError("Every retrieval text must have exactly one owner")
        result[name] = {"name": name, "manifest": manifest, "metadata": metadata, "features": features,
                        "hashes": hashes, "counts": counts, "paths": {k: str(p.relative_to(repository)) for k, p in paths.items()}}
    return result


def score_state(adapter, dataset, block_size=128):
    image_features, text_features = adapted_features(dataset["features"], adapter)
    args = (image_features, text_features, dataset["features"]["image_ids"], dataset["features"]["text_ids"])
    name, manifest = dataset["name"], dataset["manifest"]
    if name == "visual_entailment":
        metrics, predictions = evaluate_relations(*args, manifest["pairs"])
        if metrics["excluded_images"]:
            raise ValueError("Every original relation-test image must remain eligible")
    elif name in ("e_vil_test1000", "coco_karpathy"):
        metrics, predictions = compact_retrieval(*args, manifest["pairs"], block_size=block_size)
    else:
        metrics, predictions = evaluate_triplets(*args, manifest["triplets"])
    return metrics, predictions


def validate_manifest(repository, manifest_path, protocol_hash):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_sha256") != protocol_hash or manifest.get("encoder") not in ENCODERS:
        raise ValueError("Manifest protocol or encoder mismatch")
    if manifest.get("test_outcomes_used_for_selection") is not False or manifest.get("matched_controls_complete") is not True:
        raise ValueError("Scoring requires development-only selections and completed matched controls")
    for label in ("primary", "sensitivity"):
        path = manifest_path.parent / f"selection_{label}.json"
        if sha256(path) != manifest["selection_sha256"][label]:
            raise ValueError("Development selection changed")
        selection = json.loads(path.read_text())
        if selection.get("test_outcomes_used") is not False or selection["ledger_sha256"] != manifest["ledger_sha256"]:
            raise ValueError("Invalid development-selection provenance")
        if set(selection["families"]) != set(FAMILIES):
            raise ValueError("Incomplete six-family development selection")
    states = {row["state_id"]: row for row in manifest["states"]}
    for label, tolerance in (("primary", 1.), ("sensitivity", 0.)):
        selection = json.loads((manifest_path.parent / f"selection_{label}.json").read_text())
        expected_selection = set()
        for family, record in selection["families"].items():
            for run in record["runs"]:
                state = states.get(run["state_id"])
                if state is None or any(run.get(key) != state.get(key) for key in
                    ("method", "seed", "epoch", "learning_rate", "checkpoint_sha256", "source_mix", "beta", "alpha", "update_norm")):
                    raise ValueError("Manifest state differs from locked development selection")
                expected_selection.add((run["state_id"], family, run["seed"]))
        actual_selection = [(r["state_id"], r["family"], r["seed"]) for r in manifest["selections"]
                            if r["tolerance_pp"] == tolerance and r["family"] in FAMILIES]
        if len(actual_selection) != len(expected_selection) or set(actual_selection) != expected_selection:
            raise ValueError("Manifest selections differ from development-selection file")
    planned_states(manifest)
    return manifest


def make_selection_lock(repository, manifest_paths, protocol_hash):
    records = {}
    for value in manifest_paths:
        path = path_at_root(repository, value)
        manifest = validate_manifest(repository, path, protocol_hash)
        encoder = manifest["encoder"]
        if encoder in records:
            raise ValueError("Duplicate encoder in selection lock")
        records[encoder] = {"manifest": str(path.relative_to(repository)), "manifest_sha256": sha256(path),
                            "selection_sha256": manifest["selection_sha256"], "ledger_sha256": manifest["ledger_sha256"]}
    if set(records) != set(ENCODERS):
        raise ValueError("Both encoders must be locked before held-out scoring")
    return {"schema_version": 1, "protocol_sha256": protocol_hash, "encoders": records,
            "purpose": "lock all development selections before any held-out scoring", "evidence_type": EVIDENCE_TYPE}


def planned_states(manifest):
    states = {row["state_id"]: row for row in manifest["states"]}
    if len(states) != len(manifest["states"]):
        raise ValueError("Duplicate state IDs")
    selected_ids = {row["state_id"] for row in manifest["selections"]}
    # The executor exports these cells independently of primary family selection.
    decomposition = manifest.get("decomposition_selections", [])
    if len(decomposition) != 12:
        raise ValueError("Manifest must export all four decomposition cells for all three seeds")
    selected_ids.update(row["state_id"] for row in decomposition)
    if not selected_ids.issubset(states):
        raise ValueError("Selection references an unknown state")
    selections = manifest["selections"]
    for tolerance in (1., 0.):
        expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
        actual = [(r["family"], r["seed"]) for r in selections if r["tolerance_pp"] == tolerance and r["family"] in FAMILIES]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Selections must cover all six families and three seeds once")
    controls = [r for r in selections if r["family"] == "matched_allocation_distillation"]
    if len(controls) != 9 or {(r["seed"], r["draw_id"]) for r in controls} != {(s,d) for s in SEEDS for d in range(3)}:
        raise ValueError("Matched AD controls must contain all nine seed-by-draw states")
    primary_ad = {row["seed"]: states[row["state_id"]] for row in selections
                  if row["tolerance_pp"] == 1. and row["family"] == "allocation_distillation"}
    cell_names = {"U": "supported", "A": "allocation", "D": "distilled", "AD": "allocation_distillation"}
    cells_seen = []
    for row in decomposition:
        cell = row["cell"]
        if cell not in cell_names.values() or row["seed"] not in SEEDS:
            raise ValueError("Unknown decomposition cell or seed")
        state, joint = states[row["state_id"]], primary_ad[row["seed"]]
        mix = joint["source_mix"] if cell in ("allocation", "allocation_distillation") else 0.
        beta = joint["beta"] if cell in ("distilled", "allocation_distillation") else 0.
        if (any(state[key] != joint[key] for key in ("seed", "epoch", "learning_rate")) or
            state["source_mix"] != mix or state["beta"] != beta or state["family"] != cell or state["draw_id"] is not None):
            raise ValueError("Decomposition state differs from the AD-selected schedule or parameters")
        cells_seen.append((cell, row["seed"]))
    if set(cells_seen) != {(cell,seed) for cell in cell_names.values() for seed in SEEDS}:
        raise ValueError("Incomplete decomposition seed-by-cell plan")
    for row in controls:
        state, joint = states[row["state_id"]], primary_ad[row["seed"]]
        if (any(state[key] != joint[key] for key in ("seed", "epoch", "learning_rate", "source_mix", "beta")) or
            state["draw_id"] != row["draw_id"] or state["family"] != "matched_allocation_distillation"):
            raise ValueError("Random control differs from the primary AD schedule or parameters")
    frozen = {"state_id": "frozen", "method": "frozen", "family": "frozen", "seed": None,
              "epoch": 0, "learning_rate": None, "source_mix": None, "beta": None, "alpha": 0.,
              "draw_id": None, "update_norm": 0., "checkpoint": None, "checkpoint_sha256": None,
              "encoder": manifest["encoder"], "architecture": "linear"}
    return [frozen] + [states[sid] for sid in sorted(selected_ids)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--selection-lock-sha256")
    parser.add_argument("--lock-manifests", nargs=2, type=Path, help="Create lock only; do not score")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    args.protocol = path_at_root(ROOT, args.protocol)
    args.selection_lock = path_at_root(ROOT, args.selection_lock)
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Explicitly frozen protocol digest mismatch")
    validate_evaluation_specification(json.loads(args.protocol.read_text()))
    if args.lock_manifests:
        lock = make_selection_lock(ROOT, args.lock_manifests, args.protocol_sha256)
        if args.selection_lock.exists() and json.loads(args.selection_lock.read_text()) != lock:
            raise ValueError("Existing selection lock is immutable")
        write_json(args.selection_lock, lock)
        print(json.dumps({"selection_lock": str(args.selection_lock), "sha256": sha256(args.selection_lock)}))
        return
    if any(value is None for value in (args.manifest, args.dataset_config, args.output, args.selection_lock_sha256)):
        parser.error("Scoring requires --manifest, --dataset-config, --output and --selection-lock-sha256")
    for field in ("manifest", "dataset_config", "output"):
        setattr(args, field, path_at_root(ROOT, getattr(args, field)))
    if sha256(args.selection_lock) != args.selection_lock_sha256:
        raise ValueError("Selection-lock digest mismatch")
    lock = json.loads(args.selection_lock.read_text())
    reconstructed = make_selection_lock(ROOT, [record["manifest"] for record in lock["encoders"].values()], args.protocol_sha256)
    if lock != reconstructed:
        raise ValueError("A manifest or development selection changed after the cross-encoder lock")
    manifest = validate_manifest(ROOT, args.manifest, args.protocol_sha256)
    if sha256(args.manifest) != lock["encoders"][manifest["encoder"]]["manifest_sha256"]:
        raise ValueError("Requested manifest is not the locked encoder manifest")
    ledger_path = args.manifest.parent / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    from gcr.training import canonical_json
    import hashlib
    if hashlib.sha256(canonical_json(ledger["identity"])).hexdigest() != ledger["ledger_sha256"] or ledger["ledger_sha256"] != manifest["ledger_sha256"]:
        raise ValueError("Execution ledger content hash mismatch")
    for name, digest in ledger["identity"]["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Training implementation changed after execution: {name}")
    runs = planned_states(manifest)
    for run in runs:
        if Path(run["state_id"]).name != run["state_id"] or run["state_id"] in (".", ".."):
            raise ValueError("Unsafe state identifier")
        if run["checkpoint"] is not None and sha256(path_at_root(ROOT, run["checkpoint"])) != run["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint content hash mismatch")
    datasets = load_dataset_config(ROOT, args.dataset_config, manifest["encoder"], ledger)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    source_hashes = {name: sha256(ROOT / name) for name in SOURCE_PATHS}
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "torch_threads": torch.get_num_threads(), "blas_thread_environment": {name: os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    receipt = {"schema_version": 1, "protocol_sha256": args.protocol_sha256, "encoder": manifest["encoder"],
        "manifest_sha256": sha256(args.manifest), "ledger_file_sha256": sha256(ledger_path),
        "selection_lock_sha256": args.selection_lock_sha256, "dataset_config_sha256": sha256(args.dataset_config),
        "source_hashes": source_hashes, "input_hashes": {name: data["hashes"] for name,data in datasets.items()},
        "runs": runs, "selections": manifest["selections"], "decomposition_selections": manifest["decomposition_selections"],
        "environment": environment, "block_size": args.block_size, "evidence_type": EVIDENCE_TYPE}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "prescore_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Existing pre-score receipt is immutable; use a new output directory")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    index = {**{key: receipt[key] for key in ("schema_version", "protocol_sha256", "encoder", "manifest_sha256", "selection_lock_sha256", "source_hashes", "selections", "decomposition_selections", "evidence_type")},
        "status": "running", "manifest": str(args.manifest.relative_to(ROOT)), "datasets": list(DATASETS),
        "prescore_receipt": str(receipt_path.relative_to(ROOT)), "prescore_receipt_sha256": sha256(receipt_path),
        "normalization": "same final float32 L2 normalization for frozen and adapted features",
        "score_precision": "unscaled float64 dot products", "retrieval_tie_policy": RETRIEVAL_TIE_POLICY,
        "pair_tie_policy": PAIR_TIE_POLICY, "runs": []}
    for run in runs:
        adapter = None
        if run["checkpoint"]:
            payload = torch.load(ROOT / run["checkpoint"], map_location="cpu", weights_only=True)
            for key in ("method", "seed", "epoch", "learning_rate"):
                if payload[key] != run[key]:
                    raise ValueError(f"Checkpoint state metadata mismatch: {run['state_id']} {key}")
            if payload.get("protocol_sha256") != args.protocol_sha256 or payload.get("ledger_sha256") != manifest["ledger_sha256"]:
                raise ValueError("Checkpoint protocol or execution-ledger binding mismatch")
            adapter = ResidualAdapter(512 if manifest["encoder"] == "vit_b32" else 1024)
            adapter.load_state_dict(payload["state_dict"], strict=True)
            adapter.eval()
            actual_norm = sum(float(v.double().square().sum()) for v in adapter.state_dict().values()) ** .5
            if not np.isclose(actual_norm, run["update_norm"], atol=1e-12, rtol=1e-12):
                raise ValueError("Recorded learned update norm differs from checkpoint parameters")
        output_run = dict(run, datasets={})
        for name, dataset in datasets.items():
            started = time.monotonic()
            directory = args.output / "predictions" / run["state_id"]
            directory.mkdir(parents=True, exist_ok=True)
            archive, metadata_path = directory / f"{name}.npz", directory / f"{name}.json"
            provenance = {"run": run, "dataset": name, "prescore_receipt_sha256": sha256(receipt_path)}
            reused = False
            if archive.exists() or metadata_path.exists():
                if not archive.exists() or not metadata_path.exists():
                    raise ValueError("Incomplete prediction pair; inspect before resuming")
                metadata = json.loads(metadata_path.read_text())
                if metadata["provenance"] != provenance or metadata["predictions_sha256"] != sha256(archive):
                    raise ValueError("Existing predictions have different content or provenance")
                metrics = metadata["metrics"]
                reused = True
            else:
                metrics, predictions = score_state(adapter, dataset, args.block_size)
                temporary = archive.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **predictions)
                temporary.replace(archive)
                write_json(metadata_path, {"metrics": metrics, "provenance": provenance,
                    "predictions_sha256": sha256(archive), "seconds": time.monotonic() - started})
            output_run["datasets"][name] = {"metrics": metrics, "predictions": str(archive.relative_to(ROOT)),
                "predictions_sha256": sha256(archive), "metadata": str(metadata_path.relative_to(ROOT)),
                "metadata_sha256": sha256(metadata_path)}
            print(json.dumps({"state": run["state_id"], "dataset": name, "reused": reused,
                              "seconds": round(time.monotonic() - started, 3)}), flush=True)
        index["runs"].append(output_run)
        write_json(args.output / "index.json", index)
    index["status"] = "complete"
    index["run_count"] = len(index["runs"])
    write_json(args.output / "index.json", index)


if __name__ == "__main__":
    main()
