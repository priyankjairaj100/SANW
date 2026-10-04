#!/usr/bin/env python3
"""Locked, exploratory practical-v6 evaluation without benchmark-dependent routing.

One fixed scalar score is applied to full retrieval galleries and SC++ pairs.
Retrieval uses an exact bounded-residual shortlist, not a fixed top-k reranker.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ENCODERS = ("vit_b32", "rn50")
SEEDS = (17, 29, 43)
DATASETS = ("e_vil_test1000", "coco_karpathy", "sugarcrepe_pp")
EXPECTED = {"e_vil_test1000": (1000, 5000), "coco_karpathy": (5000, 25000), "sugarcrepe_pp": (1542, None)}
FAMILY_SIZE, REPLICATES, BOOTSTRAP_SEED = 80, 100000, 20261007
TIE_POLICY = "descending float64 cosine plus bounded float32 residual, then ascending candidate manifest index"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path(value):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Artifact escapes repository")
    return p


def read(value):
    return json.loads(path(value).read_text())


def write(p, value):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def immutable(p, value):
    p = path(p)
    if p.exists() and read(p) != value:
        raise ValueError(f"Immutable artifact changed: {p}")
    if not p.exists():
        write(p, value)


def relative(p):
    return str(path(p).relative_to(ROOT))


def verify_manifest(p, protocol_hash):
    m = read(p)
    if m["encoder"] not in ENCODERS or m["protocol_sha256"] != protocol_hash:
        raise ValueError("Manifest encoder/protocol mismatch")
    if m.get("test_outcomes_used_for_selection") is not False or m.get("current_test_outcomes_used_for_selection") is not False:
        raise ValueError("Selection must exclude current test outcomes")
    if m.get("evaluation_status") != "exploratory_after_historical_test_reuse":
        raise ValueError("Historical test exposure must remain explicit")
    states = m["states"]
    if len(states) != 3 or sorted(s["seed"] for s in states) != list(SEEDS):
        raise ValueError("Exactly three selected seeds required")
    if len({s["state_id"] for s in states}) != 3:
        raise ValueError("Duplicate state identifiers")
    for s in states:
        if Path(s["state_id"]).name != s["state_id"] or s["state_id"] in (".", "..", "frozen"):
            raise ValueError("Unsafe state identifier")
        if not isinstance(s["epoch"], int) or s["epoch"] < 1 or s["update_norm"] <= 0:
            raise ValueError("Selected state must have a nonzero trained update")
        if digest(path(s["checkpoint"])) != s["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
    if not m.get("source_hashes"):
        raise ValueError("Training source hashes required")
    for filename, expected in m["source_hashes"].items():
        if digest(path(filename)) != expected:
            raise ValueError(f"Training source changed: {filename}")
    if digest(path(m["training_metadata"]["path"])) != m["training_metadata"]["sha256"]:
        raise ValueError("Training feature provenance changed")
    return m


def create_lock(protocol, protocol_hash, manifests):
    if digest(path(protocol)) != protocol_hash:
        raise ValueError("Protocol content changed")
    records, families = {}, set()
    for p in manifests:
        m = verify_manifest(p, protocol_hash)
        if m["encoder"] in records:
            raise ValueError("Duplicate encoder")
        records[m["encoder"]] = {"manifest": relative(p), "manifest_sha256": digest(path(p))}
        families.add(m["family"])
    if set(records) != set(ENCODERS) or len(families) != 1:
        raise ValueError("Same family and both encoders required")
    sources = ["scripts/evaluate_practical_v6.py", "scripts/audit_practical_v6.py", "src/gcr/practical_scorer.py"]
    return {"schema_version": 1, "protocol": relative(protocol), "protocol_sha256": protocol_hash,
            "encoders": records, "family": families.pop(), "source_hashes": {s: digest(ROOT / s) for s in sources},
            "evaluation_status": "exploratory_after_historical_test_reuse",
            "family_size": FAMILY_SIZE, "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_endpoints": ["e_vil_test1000:i2t.r1", "e_vil_test1000:t2i.r1", "sugarcrepe_pp:both_accuracy"],
            "secondary_endpoints": ["coco_karpathy:i2t.r1", "coco_karpathy:t2i.r1"]}


def load_lock(filename, expected):
    if digest(path(filename)) != expected:
        raise ValueError("Selection lock hash mismatch")
    lock = read(filename)
    rebuilt = create_lock(lock["protocol"], lock["protocol_sha256"], [r["manifest"] for r in lock["encoders"].values()])
    if rebuilt != lock:
        raise ValueError("A locked source, checkpoint, or manifest changed")
    return lock


def load_dataset(config, name, encoder, training_metadata):
    entry = config["datasets"][name]
    if set(entry) != {"manifest", "features", "metadata"}:
        raise ValueError("Dataset requires manifest, features, and metadata")
    hashes = {key: digest(path(value)) for key, value in entry.items()}
    manifest, meta = read(entry["manifest"]), read(entry["metadata"])
    if meta.get("manifest_sha256", meta.get("input_manifest_sha256")) != hashes["manifest"]:
        raise ValueError("Feature metadata is not bound to manifest")
    if meta.get("features_sha256") not in (None, hashes["features"]):
        raise ValueError("Feature archive differs from metadata")
    for key in ("model_revision", "weights_sha256", "open_clip_version", "logit_scale"):
        if key not in training_metadata or meta.get(key) != training_metadata[key]:
            raise ValueError(f"Encoder provenance differs: {key}")
    with np.load(path(entry["features"]), allow_pickle=False) as z:
        arrays = {k: z[k] for k in ("image_features", "text_features", "image_ids", "text_ids")}
    for kind in ("image", "text"):
        ids, features = arrays[f"{kind}_ids"], arrays[f"{kind}_features"]
        if ids.tolist() != [str(r["id"]) for r in manifest[f"{kind}s"]] or len(set(ids)) != len(ids):
            raise ValueError("Feature order or unique IDs differ from manifest")
        if features.shape != (len(ids), 512 if encoder == "vit_b32" else 1024) or not np.isfinite(features).all():
            raise ValueError("Invalid feature shape or values")
        if not np.allclose(np.linalg.norm(features, axis=1), 1., rtol=0, atol=1e-4):
            raise ValueError("Features must already be normalized")
    # Match the original evaluator: one final float32 L2 normalization.
    for kind in ("image", "text"):
        values = torch.from_numpy(np.ascontiguousarray(arrays[f"{kind}_features"], dtype=np.float32))
        arrays[f"{kind}_features"] = torch.nn.functional.normalize(values, dim=-1).numpy()
    mask = np.asarray([row["split"] == "test" for row in manifest["images"]])
    arrays["image_features"], arrays["image_ids"] = arrays["image_features"][mask], arrays["image_ids"][mask]
    ni, nt = EXPECTED[name]
    if len(arrays["image_ids"]) != ni or (nt is not None and len(arrays["text_ids"]) != nt):
        raise ValueError("Benchmark incomplete")
    if name == "sugarcrepe_pp" and len(manifest["triplets"]) != 4757:
        raise ValueError("SC++ benchmark incomplete")
    if name != "sugarcrepe_pp":
        pairs = manifest["pairs"]
        if any(p["relation"] != "source" for p in pairs):
            raise ValueError("Retrieval relevance must be source ownership")
        if Counter(str(p["image_id"]) for p in pairs) != Counter({str(i): 5 for i in arrays["image_ids"]}):
            raise ValueError("Retrieval image must own exactly five captions")
        if Counter(str(p["text_id"]) for p in pairs) != Counter({str(i): 1 for i in arrays["text_ids"]}):
            raise ValueError("Retrieval captions must have exactly one owner")
    return {"name": name, "manifest": manifest, "arrays": arrays, "hashes": hashes, "paths": entry}


def load_model(state, dimension):
    from gcr.practical_scorer import BoundedPairScorer
    checkpoint = torch.load(path(state["checkpoint"]), map_location="cpu", weights_only=True)
    for key in ("seed", "epoch"):
        if checkpoint.get(key) != state[key]:
            raise ValueError("Checkpoint header does not match selected state")
    config = checkpoint.get("model_config", checkpoint.get("config"))
    model = BoundedPairScorer(**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if config["dimension"] != dimension:
        raise ValueError("Checkpoint encoder dimension mismatch")
    if checkpoint.get("optimizer_steps", 0) <= 0 or not np.isclose(checkpoint.get("update_norm", -1), state["update_norm"], rtol=0, atol=1e-12):
        raise ValueError("Selected training update differs from checkpoint")
    if not torch.any(checkpoint["state_dict"]["output.weight"] != 0):
        raise ValueError("Selected model retains a zero output projection")
    model.eval()
    return model


def residual(model, images, texts, chunk=256):
    """Canonical fixed-shape MLP inference; identical pairs receive exact ties.

    Every pair is evaluated at row zero of a [1,D] matrix. Multi-row GEMM can
    change final ulps with row position, even when its batch shape is padded.
    Chunking only transfers input arrays; it never changes the MLP batch shape.
    """
    if model is None:
        return np.zeros(len(images), dtype=np.float64)
    result = np.empty(len(images), dtype=np.float64)
    with torch.inference_mode():
        for first in range(0, len(images), chunk):
            a = torch.from_numpy(np.asarray(images[first:first + chunk], dtype=np.float32))
            b = torch.from_numpy(np.asarray(texts[first:first + chunk], dtype=np.float32))
            for row in range(len(a)):
                result[first + row] = float(model.residual_pairs(a[row:row + 1], b[row:row + 1])[0])
    if not np.isfinite(result).all() or np.any(np.abs(result) > float(model.epsilon)):
        raise ValueError("Bounded residual is invalid")
    return result


def retrieval(model, dataset, block_size=64):
    a, manifest = dataset["arrays"], dataset["manifest"]
    images, texts = a["image_features"], a["text_features"]
    image_map = {str(i): k for k, i in enumerate(a["image_ids"])}
    text_map = {str(i): k for k, i in enumerate(a["text_ids"])}
    owners = np.empty(len(texts), dtype=np.int64)
    for pair in manifest["pairs"]:
        owners[text_map[str(pair["text_id"])]] = image_map[str(pair["image_id"])]
    raw = {"image_ids": a["image_ids"], "text_ids": a["text_ids"], "text_source_image_ids": a["image_ids"][owners]}
    eps = 0. if model is None else float(model.epsilon)
    # Conservative allowance for float64 summation and bounded float32 output.
    guard = 1e-8 + 8 * images.shape[1] * np.finfo(np.float64).eps
    metrics = {"images": len(images), "texts": len(texts), "epsilon": eps, "shortlist_guard": guard, "tie_policy": TIE_POLICY}
    for direction, queries, candidates in (("i2t", images, texts), ("t2i", texts, images)):
        winners = np.empty(len(queries), dtype=np.int64)
        winning_scores = np.empty(len(queries), dtype=np.float64)
        runner_scores = np.empty(len(queries), dtype=np.float64)
        counts = np.empty(len(queries), dtype=np.int64)
        frozen_max = np.empty(len(queries), dtype=np.float64)
        for start in range(0, len(queries), block_size):
            base = queries[start:start + block_size].astype(np.float64) @ candidates.astype(np.float64).T
            threshold = np.max(base, axis=1) - 2 * eps - guard
            rows, cols = np.nonzero(base >= threshold[:, None])
            qi = start + rows
            correction = np.empty(len(qi), dtype=np.float64)
            for first in range(0, len(qi), 8192):
                qidx, cidx = qi[first:first + 8192], cols[first:first + 8192]
                x, t = (queries[qidx], candidates[cidx]) if direction == "i2t" else (candidates[cidx], queries[qidx])
                correction[first:first + len(qidx)] = residual(model, x, t)
            scores = base[rows, cols] + correction
            offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(base)))]
            for offset in range(len(base)):
                lo, hi = offsets[offset:offset + 2]
                values, indices = scores[lo:hi], cols[lo:hi]
                order = np.lexsort((indices, -values))
                best = order[0]
                winners[start + offset], winning_scores[start + offset] = indices[best], values[best]
                runner_scores[start + offset] = values[order[1]] if len(order) > 1 else -np.inf
                counts[start + offset], frozen_max[start + offset] = hi - lo, np.max(base[offset])
        correct = owners[winners] == np.arange(len(images)) if direction == "i2t" else winners == owners
        raw.update({f"{direction}_top_indices": winners, f"{direction}_top_scores": winning_scores,
                    f"{direction}_runner_up_scores": runner_scores, f"{direction}_correct": correct,
                    f"{direction}_candidate_counts": counts, f"{direction}_frozen_max_scores": frozen_max})
        metrics[direction] = {"r1": float(correct.mean()), "scored_pairs": int(counts.sum()),
                              "total_pairs": len(queries) * len(candidates), "max_candidates": int(counts.max())}
    return metrics, raw


def triplets(model, dataset):
    a, triplets = dataset["arrays"], dataset["manifest"]["triplets"]
    image_map = {str(i): k for k, i in enumerate(a["image_ids"])}
    text_map = {str(i): k for k, i in enumerate(a["text_ids"])}
    raw = {"item_ids": np.asarray([str(r["id"]) for r in triplets]),
           "image_ids": np.asarray([str(r["image_id"]) for r in triplets]),
           "categories": np.asarray([str(r["category"]) for r in triplets])}
    images = a["image_features"][[image_map[i] for i in raw["image_ids"]]]
    for field in ("positive1", "positive2", "negative"):
        ids = np.asarray([str(r[f"{field}_id"]) for r in triplets])
        texts = a["text_features"][[text_map[i] for i in ids]]
        raw[f"{field}_ids"] = ids
        raw[f"{field}_scores"] = np.einsum("ij,ij->i", images.astype(np.float64), texts.astype(np.float64)) + residual(model, images, texts)
    raw["positive1_correct"] = raw["positive1_scores"] > raw["negative_scores"]
    raw["positive2_correct"] = raw["positive2_scores"] > raw["negative_scores"]
    raw["correct"] = raw["positive1_correct"] & raw["positive2_correct"]
    metrics = {"items": len(triplets), "images": len(set(raw["image_ids"])),
               "positive1_accuracy": float(raw["positive1_correct"].mean()),
               "positive2_accuracy": float(raw["positive2_correct"].mean()),
               "both_accuracy": float(raw["correct"].mean())}
    metrics["near_tie_pairs_1e8"] = int(np.count_nonzero(np.abs(raw["positive1_scores"] - raw["negative_scores"]) <= 1e-8) + np.count_nonzero(np.abs(raw["positive2_scores"] - raw["negative_scores"]) <= 1e-8))
    return metrics, raw


def evaluate(args):
    lock = load_lock(args.selection_lock, args.selection_lock_sha256)
    manifest_record = lock["encoders"][args.encoder]
    manifest = read(manifest_record["manifest"])
    config = read(args.dataset_config)
    if config["encoder"] != args.encoder:
        raise ValueError("Encoder dataset config mismatch")
    training_meta = read(manifest["training_metadata"]["path"])
    datasets = {name: load_dataset(config, name, args.encoder, training_meta) for name in DATASETS}
    states = [{"state_id": "frozen", "seed": None, "epoch": 0, "update_norm": 0., "checkpoint": None}] + manifest["states"]
    receipt = {"schema_version": 1, "encoder": args.encoder, "selection_lock": relative(args.selection_lock),
               "selection_lock_sha256": args.selection_lock_sha256, "manifest": manifest_record["manifest"],
               "manifest_sha256": manifest_record["manifest_sha256"], "dataset_config": relative(args.dataset_config),
               "dataset_config_sha256": digest(path(args.dataset_config)), "states": states,
               "input_hashes": {name: d["hashes"] for name, d in datasets.items()}, "family": lock["family"],
               "source_hashes": lock["source_hashes"], "evaluation_status": lock["evaluation_status"],
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
               "score": "float64 frozen cosine plus bounded float32 residual; one score for all datasets",
               "block_size": args.block_size, "torch_threads": args.torch_threads,
               "residual_inference": "Each pair uses row zero of a [1,D] float32 MLP call; one Torch thread; chunks only transfer arrays."}
    out = path(args.output)
    immutable(out / "prescore_receipt.json", receipt)
    index = {"schema_version": 1, "status": "running", "encoder": args.encoder,
             "prescore_receipt": relative(out / "prescore_receipt.json"),
             "prescore_receipt_sha256": digest(out / "prescore_receipt.json"), "runs": []}
    for state in states:
        model = None if state["checkpoint"] is None else load_model(state, 512 if args.encoder == "vit_b32" else 1024)
        record = {"state": state, "datasets": {}}
        for name, dataset in datasets.items():
            started = time.monotonic()
            archive = out / "predictions" / state["state_id"] / f"{name}.npz"
            metadata = archive.with_suffix(".json")
            provenance = {"state": state, "dataset": name, "prescore_receipt_sha256": index["prescore_receipt_sha256"]}
            if archive.exists() or metadata.exists():
                saved = read(metadata)
                if saved["provenance"] != provenance or saved["predictions_sha256"] != digest(archive):
                    raise ValueError("Saved predictions changed")
            else:
                metrics, raw = triplets(model, dataset) if name == "sugarcrepe_pp" else retrieval(model, dataset, args.block_size)
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, **raw)
                saved = {"provenance": provenance, "metrics": metrics, "predictions_sha256": digest(archive)}
                write(metadata, saved)
            record["datasets"][name] = {"predictions": relative(archive), "predictions_sha256": digest(archive),
                                         "metadata": relative(metadata), "metadata_sha256": digest(metadata), "metrics": saved["metrics"]}
            print(json.dumps({"state": state["state_id"], "dataset": name, "seconds": time.monotonic() - started}), flush=True)
        index["runs"].append(record)
        write(out / "index.json", index)
    index["status"] = "complete"
    write(out / "index.json", index)


def bootstrap(delta, clusters):
    _, inverse = np.unique(clusters, return_inverse=True)
    averaged = delta.mean(axis=0)
    sums = np.bincount(inverse, weights=averaged)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(REPLICATES)
    for start in range(0, REPLICATES, 250):
        draws = rng.integers(0, len(counts), size=(min(250, REPLICATES - start), len(counts)))
        samples[start:start + len(draws)] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = .05 / (2 * FAMILY_SIZE)
    lower, upper = np.quantile(samples, [tail, 1 - tail], method="linear")
    return {"difference": float(averaged.mean()), "ci_lower": float(lower), "ci_upper": float(upper),
            "seed_differences": delta.mean(axis=1).tolist(), "family_size": FAMILY_SIZE,
            "bootstrap_seed": BOOTSTRAP_SEED, "replicates": REPLICATES,
            "images": len(counts), "items": len(clusters), "conditioning": "fixed selected seed mean; image clusters resampled"}, samples


def analyze(args):
    effects, draws, indices, family, lock_hash = [], {}, {}, None, None
    for filename in args.indices:
        index = read(filename)
        if index["status"] != "complete" or index["encoder"] in indices:
            raise ValueError("Complete unique encoder indices required")
        receipt = read(index["prescore_receipt"])
        if digest(path(index["prescore_receipt"])) != index["prescore_receipt_sha256"]:
            raise ValueError("Prescore receipt changed")
        lock = load_lock(receipt["selection_lock"], receipt["selection_lock_sha256"])
        if family is not None and (family != lock["family"] or lock_hash != receipt["selection_lock_sha256"]):
            raise ValueError("Indices must share the family and selection lock")
        family, lock_hash = lock["family"], receipt["selection_lock_sha256"]
        indices[index["encoder"]] = {"path": relative(filename), "sha256": digest(path(filename))}
        state_runs = sorted([r for r in index["runs"] if r["state"]["seed"] is not None], key=lambda r: r["state"]["seed"])
        frozen = [r for r in index["runs"] if r["state"]["state_id"] == "frozen"]
        if len(frozen) != 1 or [r["state"]["seed"] for r in state_runs] != list(SEEDS):
            raise ValueError("Incomplete selected seed grid")
        for name in DATASETS:
            raws = []
            for run in [frozen[0]] + state_runs:
                artifact = run["datasets"][name]
                if digest(path(artifact["predictions"])) != artifact["predictions_sha256"]:
                    raise ValueError("Prediction archive changed")
                with np.load(path(artifact["predictions"]), allow_pickle=False) as z:
                    raws.append({k: z[k] for k in z.files})
            for metric in (("both_accuracy",) if name == "sugarcrepe_pp" else ("i2t.r1", "t2i.r1")):
                field = "correct" if name == "sugarcrepe_pp" else metric.split(".")[0] + "_correct"
                ids_key = "item_ids" if name == "sugarcrepe_pp" else "image_ids" if metric == "i2t.r1" else "text_ids"
                clusters_key = "text_source_image_ids" if metric == "t2i.r1" else "image_ids"
                if any(not np.array_equal(r[ids_key], raws[0][ids_key]) or not np.array_equal(r[clusters_key], raws[0][clusters_key]) for r in raws[1:]):
                    raise ValueError("Paired prediction identities differ")
                delta = np.stack([r[field].astype(float) - raws[0][field].astype(float) for r in raws[1:]])
                effect, samples = bootstrap(delta, raws[0][clusters_key])
                eid = f"{index['encoder']}__{name}__{metric.replace('.', '_')}"
                effect.update(effect_id=eid, encoder=index["encoder"], dataset=name, metric=metric,
                              frozen=float(raws[0][field].mean()), selected=float(np.stack([r[field] for r in raws[1:]]).mean()))
                effects.append(effect)
                draws[eid] = samples
    if set(indices) != set(ENCODERS):
        raise ValueError("Both encoder indices required")
    gates = []
    for encoder in ENCODERS:
        own = {(r["dataset"], r["metric"]): r for r in effects if r["encoder"] == encoder}
        retrieval_ok = all(own[("e_vil_test1000", m)]["ci_lower"] > -.01 for m in ("i2t.r1", "t2i.r1"))
        caption_ok = own[("sugarcrepe_pp", "both_accuracy")]["ci_lower"] > 0
        coco_ok = all(own[("coco_karpathy", m)]["ci_lower"] > -.01 for m in ("i2t.r1", "t2i.r1"))
        gates.append({"encoder": encoder, "nonzero_three_seeds": True, "retrieval_retention": retrieval_ok,
                      "caption_improvement": caption_ok, "passed": retrieval_ok and caption_ok,
                      "secondary_coco_retention": coco_ok})
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "bootstrap_samples.npz", **draws)
    report = {"schema_version": 1, "family": family, "selection_lock_sha256": lock_hash, "indices": indices,
              "effects": effects, "gates": gates, "cross_encoder_passed": all(g["passed"] for g in gates),
              "cross_encoder_with_coco_passed": all(g["passed"] and g["secondary_coco_retention"] for g in gates),
              "evaluation_status": "exploratory_after_historical_test_reuse", "alpha": .05,
              "bootstrap_samples": relative(output / "bootstrap_samples.npz"),
              "bootstrap_samples_sha256": digest(output / "bootstrap_samples.npz")}
    write(output / "analysis.json", report)
    print(json.dumps({"analysis": relative(output / "analysis.json"), "gates": gates}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--protocol", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--manifests", nargs=2, required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--selection-lock", required=True)
    p.add_argument("--selection-lock-sha256", required=True)
    p.add_argument("--encoder", choices=ENCODERS, required=True)
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--torch-threads", type=int, choices=[1], default=1)
    p = sub.add_parser("analyze")
    p.add_argument("--indices", nargs=2, required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "lock":
        immutable(args.output, create_lock(args.protocol, args.protocol_sha256, args.manifests))
        print(json.dumps({"selection_lock": relative(args.output), "sha256": digest(path(args.output))}))
    elif args.command == "evaluate":
        torch.set_num_threads(args.torch_threads)
        torch.use_deterministic_algorithms(True)
        evaluate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
