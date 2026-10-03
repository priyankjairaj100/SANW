#!/usr/bin/env python3
"""Evaluate frozen review-follow-up state plans without changing the original study.

Every invocation writes an immutable pre-scoring receipt. Reuse requires identical
checkpoint, data and source hashes. Terminal, trajectory and selected suites can
share compact prediction archives; no checkpoint is chosen from test results.
"""
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
from gcr.evaluation import (PAIR_TIE_POLICY, RETRIEVAL_TIE_POLICY, evaluate_relations,
                            evaluate_triplets, ranks_to_metrics)
from evaluate_study import adapted_features, load_dataset, sha256, write_json

POLICIES = ("source", "supported") + tuple(f"{kind}_draw_{draw}" for kind in ("count_only", "score_stratified") for draw in range(3))
SEEDS = (17, 29, 43)
LEARNING_RATES = (0.0001, 0.0003, 0.001)
DATASETS = ("e_vil_test1000", "visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
RETRIEVAL_DATASETS = frozenset({"e_vil_test1000", "coco_karpathy"})


def validate_protocol(protocol):
    checks = {
        ("training", "policies"): list(POLICIES),
        ("training", "seeds"): list(SEEDS),
        ("training", "learning_rates"): list(LEARNING_RATES),
        ("training", "epochs"): 10,
        ("evaluation", "primary_epoch"): 10,
        ("evaluation", "trajectory_epochs"): [1, 5, 10],
        ("primary_analysis", "epoch"): 10,
        ("primary_analysis", "learning_rates"): list(LEARNING_RATES),
        ("primary_analysis", "family_size"): 18,
        ("primary_analysis", "alpha"): 0.05,
        ("primary_analysis", "bootstrap_replicates"): 10000,
        ("primary_analysis", "bootstrap_seed"): 20261004,
        ("primary_analysis", "endpoints"): [{"dataset": "e_vil_test1000", "metric": metric} for metric in ("i2t.r1", "t2i.r1")],
    }
    for (section, field), expected in checks.items():
        if protocol.get(section, {}).get(field) != expected:
            raise ValueError(f"Unsupported frozen protocol field {section}.{field}")
    contrasts = protocol["primary_analysis"]["contrasts"]
    right_sides = [["source"], [f"count_only_draw_{i}" for i in range(3)], [f"score_stratified_draw_{i}" for i in range(3)]]
    if len(contrasts) != 3 or any(row["left_conditions"] != ["supported"] or row["right_conditions"] != expected
                                  for row, expected in zip(contrasts, right_sides)):
        raise ValueError("Unsupported primary contrast family")


def compact_ranks(queries, candidates, relevance, block_size=128):
    """Exact stable ranks and best relevant scores, without retaining top-k lists."""
    queries, candidates = np.asarray(queries, dtype=np.float64), np.asarray(candidates, dtype=np.float64)
    if queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]:
        raise ValueError("Invalid retrieval feature shapes")
    if not np.isfinite(queries).all() or not np.isfinite(candidates).all():
        raise ValueError("Nonfinite retrieval features")
    if len(queries) != len(relevance) or block_size < 1 or not len(candidates):
        raise ValueError("Invalid retrieval relevance or block size")
    ranks = np.empty(len(queries), dtype=np.int64)
    best_indices = np.empty(len(queries), dtype=np.int64)
    best_scores = np.empty(len(queries), dtype=np.float64)
    indices = np.arange(len(candidates))
    for start in range(0, len(queries), block_size):
        scores = queries[start:start + block_size] @ candidates.T
        for offset, row in enumerate(scores):
            qi = start + offset
            relevant = np.unique(np.asarray(relevance[qi], dtype=np.int64))
            if not len(relevant) or relevant[0] < 0 or relevant[-1] >= len(candidates):
                raise ValueError(f"Invalid relevance for query {qi}")
            best = int(relevant[np.argmax(row[relevant])])
            value = row[best]
            ranks[qi] = 1 + np.count_nonzero(row > value) + np.count_nonzero((row == value) & (indices < best))
            best_indices[qi], best_scores[qi] = best, value
    return {"ranks": ranks, "best_relevant_indices": best_indices, "best_relevant_scores": best_scores}


def compact_retrieval(image_features, text_features, image_ids, text_ids, pairs, block_size=128):
    image_lookup, text_lookup = ({str(value): i for i, value in enumerate(ids)} for ids in (image_ids, text_ids))
    if len(image_lookup) != len(image_ids) or len(text_lookup) != len(text_ids):
        raise ValueError("Duplicate retrieval identifiers")
    image_relevance, text_relevance = [[] for _ in image_ids], [[] for _ in text_ids]
    for pair in pairs:
        if pair["relation"] != "source":
            continue
        ii, ti = image_lookup[str(pair["image_id"])], text_lookup[str(pair["text_id"])]
        image_relevance[ii].append(ti)
        text_relevance[ti].append(ii)
    if any(len(set(row)) != 1 for row in text_relevance):
        raise ValueError("Each source text must have exactly one owning image")
    i2t = compact_ranks(image_features, text_features, image_relevance, block_size)
    t2i = compact_ranks(text_features, image_features, text_relevance, block_size)
    predictions = {"image_ids": np.asarray(image_ids, dtype=str), "text_ids": np.asarray(text_ids, dtype=str),
                   "text_source_image_ids": np.asarray([str(image_ids[row[0]]) for row in text_relevance], dtype=str)}
    for direction, values in (("i2t", i2t), ("t2i", t2i)):
        predictions.update({f"{direction}_{key}": value for key, value in values.items()})
    metrics = {"images": len(image_ids), "texts": len(text_ids), "i2t": ranks_to_metrics(i2t["ranks"]),
               "t2i": ranks_to_metrics(t2i["ranks"]), "tie_policy": RETRIEVAL_TIE_POLICY}
    metrics["mean_bidirectional_r1"] = 0.5 * (metrics["i2t"]["r1"] + metrics["t2i"]["r1"])
    return metrics, predictions


def load_followup_dataset(name):
    if name != "e_vil_test1000":
        return load_dataset(name, allow_prototype=False)
    manifest_path = ROOT / "data/review_followup/e_vil_test1000/manifest.json"
    feature_path = ROOT / "results/review_followup/features/e_vil_test1000/features.npz"
    metadata_path = feature_path.with_name("metadata.json")
    manifest, metadata = json.loads(manifest_path.read_text()), json.loads(metadata_path.read_text())
    manifest_hash = sha256(manifest_path)
    if metadata.get("manifest_sha256", metadata.get("input_manifest_sha256")) != manifest_hash:
        raise ValueError("Stale e-ViL retrieval feature cache")
    with np.load(feature_path, allow_pickle=False) as raw:
        features = {key: raw[key] for key in ("image_features", "text_features", "image_ids", "text_ids")}
    for kind, count in (("image", 1000), ("text", 5000)):
        if features[f"{kind}_ids"].tolist() != [str(row["id"]) for row in manifest[f"{kind}s"]]:
            raise ValueError(f"e-ViL {kind} feature order differs from manifest")
        values = features[f"{kind}_features"]
        if values.shape != (count, 512) or not np.isfinite(values).all():
            raise ValueError(f"Invalid complete e-ViL {kind} feature matrix")
        if not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=1e-4):
            raise ValueError("e-ViL features must be unit normalized")
    if any(row.get("split") != "test" for row in manifest["images"]):
        raise ValueError("Full e-ViL retrieval manifest must contain only test images")
    if len(manifest["pairs"]) != 5000 or any(p["relation"] != "source" for p in manifest["pairs"]):
        raise ValueError("Full e-ViL retrieval requires exactly 5000 source pairs")
    by_image = Counter(str(pair["image_id"]) for pair in manifest["pairs"])
    by_text = Counter(str(pair["text_id"]) for pair in manifest["pairs"])
    if by_image != Counter({str(iid): 5 for iid in features["image_ids"]}) or by_text != Counter({str(tid): 1 for tid in features["text_ids"]}):
        raise ValueError("e-ViL source ownership must assign exactly five unique caption IDs per image")
    return {"manifest": manifest, "features": features, "metadata": metadata,
            "counts": {"images": 1000, "texts": 5000},
            "hashes": {"manifest_sha256": manifest_hash, "features_sha256": sha256(feature_path),
                       "metadata_sha256": sha256(metadata_path)}}


def path_at_root(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def normalized_state(state):
    result = dict(state)
    for key in ("state_id", "condition", "method", "draw_id", "learning_rate", "seed", "epoch", "checkpoint", "checkpoint_sha256"):
        if key not in result:
            raise ValueError(f"State lacks {key}: {result}")
    method = result["method"]
    if method not in POLICIES:
        raise ValueError("Unplanned follow-up policy")
    expected_condition = method.split("_draw_")[0]
    expected_draw = int(method[-1]) if "_draw_" in method else None
    if result["condition"] != expected_condition or result["draw_id"] != expected_draw:
        raise ValueError("Policy condition/draw identity mismatch")
    if "assignment_seed" in result and result["assignment_seed"] != (None if expected_draw is None else (101, 211, 307)[expected_draw]):
        raise ValueError("Assignment RNG identity mismatch")
    sid = str(result["state_id"])
    if not sid or Path(sid).name != sid or sid in (".", ".."):
        raise ValueError("State identifiers must be safe single path components")
    path = path_at_root(result["checkpoint"])
    if sha256(path) != result["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    result["checkpoint"] = str(path.relative_to(ROOT))
    return result


def plan_states(manifest, suite, epochs):
    states = [normalized_state(state) for state in manifest["states"]]
    by_id = {state["state_id"]: state for state in states}
    if len(by_id) != len(states):
        raise ValueError("Duplicate checkpoint state IDs")
    expected = {(method, lr, seed, epoch) for method in POLICIES for lr in LEARNING_RATES for seed in SEEDS for epoch in range(11)}
    actual = [(state["method"], state["learning_rate"], state["seed"], state["epoch"]) for state in states]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Checkpoint manifest must cover exactly all 792 planned states")
    selections = manifest.get("selections", [])
    for selection in selections:
        if selection["state_id"] not in by_id:
            raise ValueError("Selection references an unknown checkpoint")
        state = by_id[selection["state_id"]]
        for key in ("condition", "draw_id", "learning_rate", "seed", "epoch"):
            if selection[key] != state[key]:
                raise ValueError(f"Selection metadata disagrees with checkpoint: {key}")
    selection_keys = [(row["selector"], by_id[row["state_id"]]["method"], row["seed"]) for row in selections]
    if len(selection_keys) != len(set(selection_keys)) or set(selection_keys) != {(selector, method, seed) for selector in ("native", "source_retrieval") for method in POLICIES for seed in SEEDS}:
        raise ValueError("Manifest must contain both complete independently selected strategies")
    if suite == "terminal":
        chosen = [state for state in states if state["epoch"] == 10]
    elif suite == "trajectory":
        chosen = [state for state in states if state["epoch"] in epochs]
    elif suite == "selected":
        selected_ids = {selection["state_id"] for selection in selections}
        if not selections:
            raise ValueError("Selected suite requires frozen selections")
        chosen = [state for state in states if state["state_id"] in selected_ids]
    else:
        chosen = states
    if not chosen:
        raise ValueError("Empty evaluation suite")
    frozen = {"state_id": "frozen", "condition": "frozen", "method": "frozen", "draw_id": None,
              "learning_rate": None, "seed": None, "epoch": 0, "checkpoint": None, "checkpoint_sha256": None}
    return [frozen] + sorted(chosen, key=lambda s: (s["epoch"], s["condition"], s["learning_rate"], s["seed"])), selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/review_followup/state_manifest.json")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True, help="Explicit frozen protocol digest; no implicit approval or amendment")
    parser.add_argument("--suite", choices=("terminal", "trajectory", "selected"), required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results/review_followup/evaluation")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    args.manifest = path_at_root(args.manifest)
    args.protocol = path_at_root(args.protocol)
    args.output = path_at_root(args.output)
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Protocol digest differs from the explicitly frozen digest")
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    expected_datasets = set(DATASETS) - ({"coco_karpathy"} if args.suite == "trajectory" else set())
    if args.datasets is None:
        args.datasets = [name for name in DATASETS if name in expected_datasets]
    if set(args.datasets) != expected_datasets or len(args.datasets) != len(expected_datasets):
        raise ValueError(f"Suite {args.suite} requires exactly datasets {sorted(expected_datasets)}")
    if args.epochs != protocol["evaluation"]["trajectory_epochs"]:
        raise ValueError("Trajectory epochs differ from frozen protocol")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol_sha256") != args.protocol_sha256:
        raise ValueError("Checkpoint manifest is not bound to this frozen protocol")
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    runs, selections = plan_states(manifest, args.suite, args.epochs)
    datasets = {name: load_followup_dataset(name) for name in args.datasets}
    sources = [Path(__file__).resolve(), ROOT / "scripts/evaluate_study.py", ROOT / "scripts/analyze_review_followup.py", ROOT / "scripts/analyze_study.py", ROOT / "src/gcr/evaluation.py", ROOT / "src/gcr/adapters.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "torch_threads": torch.get_num_threads(), "blas_thread_environment":
                   {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    receipt = {"schema_version": 1, "suite": args.suite, "protocol_sha256": args.protocol_sha256,
               "state_manifest_sha256": sha256(args.manifest), "source_hashes": source_hashes,
               "input_hashes": {name: value["hashes"] for name, value in datasets.items()},
               "states": runs, "selections": selections, "environment": environment,
               "block_size": args.block_size, "evidence_type": "review_followup_new_execution"}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / f"{args.suite}_prescore_receipt.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("Pre-score plan differs from frozen receipt; use a separate suite/output")
    else:
        write_json(receipt_path, receipt)
    receipt_hash = sha256(receipt_path)
    index = {"schema_version": 1, "status": "running", "suite": args.suite,
             "protocol_sha256": args.protocol_sha256, "state_manifest_sha256": sha256(args.manifest),
             "prescore_receipt": str(receipt_path.relative_to(ROOT)), "prescore_receipt_sha256": receipt_hash,
             "source_hashes": source_hashes, "datasets": list(datasets), "selections": selections, "runs": [],
             "evidence_type": receipt["evidence_type"], "retrieval_tie_policy": RETRIEVAL_TIE_POLICY,
             "pair_tie_policy": PAIR_TIE_POLICY, "normalization": "final float32 L2 normalization for adapted and frozen features",
             "score_precision": "unscaled float64 dot products"}
    for run in runs:
        adapter = None
        if run["checkpoint"] is not None:
            payload = torch.load(ROOT / run["checkpoint"], map_location="cpu", weights_only=True)
            for key in ("method", "seed", "epoch", "learning_rate", "condition", "draw_id"):
                if payload[key] != run[key]:
                    raise ValueError(f"Checkpoint metadata disagrees: {run['state_id']} {key}")
            if payload.get("protocol_sha256") != args.protocol_sha256:
                raise ValueError("Checkpoint protocol hash mismatch")
            if not manifest.get("ledger_sha256") or payload.get("ledger_sha256") != manifest["ledger_sha256"]:
                raise ValueError("Checkpoint execution ledger hash mismatch")
            adapter = ResidualAdapter(dim=512)
            adapter.load_state_dict(payload["state_dict"], strict=True)
            adapter.eval()
        result = dict(run, run_id=run["state_id"], datasets={})
        for name, dataset in datasets.items():
            started = time.monotonic()
            output_dir = args.output / "predictions" / run["state_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path, metrics_path = output_dir / f"{name}.npz", output_dir / f"{name}.json"
            provenance = {"state": run, "protocol_sha256": args.protocol_sha256, "dataset": name,
                          "source_hashes": source_hashes, "input_hashes": dataset["hashes"],
                          "environment": environment, "block_size": args.block_size,
                          "evidence_type": receipt["evidence_type"], "retrieval_tie_policy": RETRIEVAL_TIE_POLICY,
                          "pair_tie_policy": PAIR_TIE_POLICY}
            reused = False
            if predictions_path.exists() or metrics_path.exists():
                if not predictions_path.exists() or not metrics_path.exists():
                    raise ValueError("Incomplete existing prediction artifact; inspect before resuming")
                saved = json.loads(metrics_path.read_text())
                if saved["provenance"] != provenance or saved["predictions_sha256"] != sha256(predictions_path):
                    raise ValueError(f"Cannot reuse different prediction provenance: {metrics_path}")
                metrics, reused = saved["metrics"], True
            else:
                image_features, text_features = adapted_features(dataset["features"], adapter)
                shared = (image_features, text_features, dataset["features"]["image_ids"], dataset["features"]["text_ids"])
                if name in RETRIEVAL_DATASETS:
                    metrics, predictions = compact_retrieval(*shared, dataset["manifest"]["pairs"], args.block_size)
                elif name == "visual_entailment":
                    metrics, predictions = evaluate_relations(*shared, dataset["manifest"]["pairs"])
                    if metrics["excluded_images"]:
                        raise ValueError("Every held-out relation image must remain eligible")
                else:
                    metrics, predictions = evaluate_triplets(*shared, dataset["manifest"]["triplets"])
                temporary = predictions_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **predictions)
                temporary.replace(predictions_path)
                write_json(metrics_path, {"metrics": metrics, "provenance": provenance,
                                          "predictions_sha256": sha256(predictions_path), "seconds": time.monotonic() - started})
            result["datasets"][name] = {"metrics": metrics, "predictions": str(predictions_path.relative_to(ROOT)),
                                        "metadata": str(metrics_path.relative_to(ROOT)), "predictions_sha256": sha256(predictions_path)}
            print(json.dumps({"state": run["state_id"], "dataset": name, "reused": reused,
                              "seconds": round(time.monotonic() - started, 2), "metrics": metrics}), flush=True)
        index["runs"].append(result)
        write_json(args.output / f"{args.suite}_index.json", index)
    index.update(status="complete", run_count=len(index["runs"]))
    write_json(args.output / f"{args.suite}_index.json", index)


if __name__ == "__main__":
    main()
