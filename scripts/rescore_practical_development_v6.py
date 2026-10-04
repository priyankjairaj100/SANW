#!/usr/bin/env python3
"""Rescore existing practical-v6 development checkpoints under canonical inference.

Never trains, opens a held-out benchmark, or changes original candidate outputs.
Each residual is evaluated in a one-pair CPU batch with one Torch thread.
Frozen retrieval shortlists are cached and shared across requested checkpoints.
Partial trajectories produce explicitly provisional selections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_practical_v6 import residual, TIE_POLICY
from gcr.practical_scorer import BoundedPairScorer
from gcr.practical_training import select_development_epoch
from gcr.review_training import SourceRetrievalPool
from gcr.training import FeatureDataset, canonical_json, sha256_file


def root_path(value):
    result = Path(value)
    result = (result if result.is_absolute() else ROOT / result).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {value}")
    return result


def relative(value):
    return str(root_path(value).relative_to(ROOT))


def read(value):
    return json.loads(root_path(value).read_text())


def content_hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def immutable_json(filename, value):
    filename = root_path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.exists():
        if read(filename) != value:
            raise ValueError(f"Existing immutable output differs: {filename}")
        return
    with filename.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_npz(filename, arrays):
    filename = root_path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": relative(filename), "sha256": sha256_file(filename)}


def verify_record(record):
    filename = root_path(record["path"])
    if sha256_file(filename) != record["sha256"]:
        raise ValueError(f"Bound input changed: {filename}")
    return filename


def load_inputs(identity):
    inputs = identity["inputs"]
    paths = {name: verify_record(record) for name, record in inputs.items()}
    dimension = identity["model"]["dimension"]
    data = FeatureDataset(ROOT, paths["manifest"], paths["features"], paths["metadata"], dimension)
    pool = SourceRetrievalPool(ROOT, paths["development_manifest"], paths["development_features"],
                               paths["development_metadata"], data, dimension)
    for container in (data, pool):
        container.images = F.normalize(container.images, dim=-1)
        container.texts = F.normalize(container.texts, dim=-1)
    if len(data.split_indices["validation"]) != 100 or len(pool.images) != 900 or len(pool.texts) != 4500:
        raise ValueError("Development scope differs from the frozen 100/900 image protocol")
    return data, pool


def frozen_shortlist(images, texts, epsilon, block_size=64):
    """Cache the evaluator's inclusive exact R1 shortlist in both directions.

    The base GEMM block size and orientation match evaluate_practical_v6.py.
    A union of image/text pairs avoids duplicate residual computation.
    """
    images = np.asarray(images, dtype=np.float32)
    texts = np.asarray(texts, dtype=np.float32)
    guard = 1e-8 + 8 * images.shape[1] * np.finfo(np.float64).eps
    arrays, all_flat = {}, []
    for direction, queries, gallery in (("i2t", images, texts), ("t2i", texts, images)):
        row_parts, col_parts, base_parts, maxima = [], [], [], []
        gallery64 = gallery.astype(np.float64)
        for start in range(0, len(queries), block_size):
            base = queries[start:start + block_size].astype(np.float64) @ gallery64.T
            maximum = base.max(axis=1)
            row, col = np.nonzero(base >= maximum[:, None] - 2 * epsilon - guard)
            row_parts.append(row + start)
            col_parts.append(col)
            base_parts.append(base[row, col])
            maxima.append(maximum)
        row, col = np.concatenate(row_parts), np.concatenate(col_parts)
        arrays[f"{direction}_rows"] = row
        arrays[f"{direction}_columns"] = col
        arrays[f"{direction}_base"] = np.concatenate(base_parts)
        arrays[f"{direction}_max"] = np.concatenate(maxima)
        arrays[f"{direction}_offsets"] = np.r_[0, np.cumsum(np.bincount(row, minlength=len(queries)))]
        flat = row * len(texts) + col if direction == "i2t" else col * len(texts) + row
        all_flat.append(flat)
    arrays["flat_pairs"] = np.unique(np.concatenate(all_flat))
    for direction, flat in zip(("i2t", "t2i"), all_flat):
        arrays[f"{direction}_union_index"] = np.searchsorted(arrays["flat_pairs"], flat)
    return arrays, guard


def get_cached_shortlist(output, encoder, inputs, pool, epsilon, source_hashes):
    identity = {"schema": "canonical_practical_development_shortlist_v1", "encoder": encoder,
                "epsilon": epsilon, "block_size": 64, "inputs": inputs,
                "source_hashes": source_hashes, "preprocessing": "float32_F.normalize_once"}
    key = content_hash(identity)
    metadata_path = output / "shortlist_cache" / f"{encoder}_{key}.json"
    if metadata_path.exists():
        metadata = read(metadata_path)
        if metadata["identity"] != identity:
            raise ValueError("Cached shortlist identity differs")
        with np.load(verify_record(metadata["archive"]), allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        return arrays, metadata["guard"], {"path": relative(metadata_path), "sha256": sha256_file(metadata_path)}
    arrays, guard = frozen_shortlist(pool.images.numpy(), pool.texts.numpy(), epsilon)
    archive = save_npz(metadata_path.with_suffix(".npz"), arrays)
    immutable_json(metadata_path, {"identity": identity, "archive": archive, "guard": guard,
                                   "unique_scored_pairs": len(arrays["flat_pairs"])})
    return arrays, guard, {"path": relative(metadata_path), "sha256": sha256_file(metadata_path)}


def rescore_retrieval(model, pool, cache, guard):
    images, texts = pool.images.numpy(), pool.texts.numpy()
    flat = cache["flat_pairs"]
    correction = np.empty(len(flat), dtype=np.float64)
    # Gathering limited blocks avoids a temporary [all candidates,D] copy.
    for start in range(0, len(flat), 256):
        keys = flat[start:start + 256]
        correction[start:start + len(keys)] = residual(model, images[keys // len(texts)], texts[keys % len(texts)])
    raw = {"image_ids": np.asarray(pool.image_ids), "text_ids": np.asarray(pool.text_ids),
           "owner": pool.owner.numpy(), "union_pair_indices": flat, "union_residuals": correction}
    owner = pool.owner.numpy()
    for direction, count in (("i2t", len(images)), ("t2i", len(texts))):
        values = cache[f"{direction}_base"] + correction[cache[f"{direction}_union_index"]]
        columns, offsets = cache[f"{direction}_columns"], cache[f"{direction}_offsets"]
        winners = np.empty(count, dtype=np.int64)
        scores = np.empty(count, dtype=np.float64)
        runner = np.full(count, -np.inf, dtype=np.float64)
        for query, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            order = np.lexsort((columns[lo:hi], -values[lo:hi]))
            winners[query] = columns[lo:hi][order[0]]
            scores[query] = values[lo:hi][order[0]]
            if len(order) > 1:
                runner[query] = values[lo:hi][order[1]]
        correct = owner[winners] == np.arange(count) if direction == "i2t" else winners == owner
        raw[f"{direction}_correct"] = correct
        raw[f"{direction}_top_indices"] = winners
        raw[f"{direction}_top_scores"] = scores
        raw[f"{direction}_runner_up_scores"] = runner
        raw[f"{direction}_candidate_counts"] = np.diff(offsets)
    summary = {"i2t_r1": float(raw["i2t_correct"].mean()), "t2i_r1": float(raw["t2i_correct"].mean()),
               "image_correct_count": int(raw["i2t_correct"].sum()), "text_correct_count": int(raw["t2i_correct"].sum()),
               "image_count": len(images), "text_count": len(texts), "tie_rule": TIE_POLICY,
               "relevance": "source_caption_ownership", "shortlist_guard": guard,
               "unique_scored_pairs": len(flat), "canonical_inference": "one_pair_row_zero_torch_threads_one"}
    return summary, raw


def rescore_composition(model, data):
    images, texts = data.images.numpy(), data.texts.numpy()
    raw_i, raw_t, raw_labels, spans = [], [], [], []
    for index in data.split_indices["validation"]:
        indices = sorted(data.pairs[index])
        lo = len(raw_i)
        raw_i.extend([index] * len(indices))
        raw_t.extend(indices)
        raw_labels.extend(data.pairs[index][j] for j in indices)
        spans.append((index, lo, len(raw_i)))
    raw_i, raw_t, raw_labels = map(np.asarray, (raw_i, raw_t, raw_labels))
    scores = np.empty(len(raw_i), dtype=np.float64)
    for start in range(0, len(raw_i), 256):
        a = images[raw_i[start:start + 256]]
        b = texts[raw_t[start:start + 256]]
        scores[start:start + len(a)] = np.einsum("ij,ij->i", a.astype(np.float64), b.astype(np.float64)) + residual(model, a, b)
    image_ids, low, high, joint, source_correct, supported_correct = [], [], [], [], [], []
    paired_accuracy, paired_margin, counts = [], [], []
    for index, lo, hi in spans:
        labels, current = raw_labels[lo:hi], scores[lo:hi]
        source, supported, negative = current[labels == 1], current[labels == 2], current[labels == 3]
        if not len(source) or not len(supported) or not len(negative):
            continue
        margins = np.minimum(source[:, None, None], supported[None, :, None]) - negative[None, None, :]
        paired_accuracy.append(float((margins > 0).mean()))
        paired_margin.append(float(margins.mean()))
        counts.append(margins.size)
        image_ids.append(data.image_ids[index])
        low.append(float(min(source.min(), supported.min())))
        high.append(float(negative.max()))
        joint.append(low[-1] > high[-1])
        source_correct.append(bool(source.min() > high[-1]))
        supported_correct.append(bool(supported.min() > high[-1]))
    if len(image_ids) != 100:
        raise ValueError("Canonical composition must cover all 100 development images")
    summary = {"image_count": len(image_ids), "all_pairs_correct_count": int(sum(joint)),
               "all_pairs_accuracy": float(np.mean(joint)), "source_all_correct_count": int(sum(source_correct)),
               "supported_all_correct_count": int(sum(supported_correct)), "paired_joint_accuracy": float(np.mean(paired_accuracy)),
               "mean_paired_joint_margin": float(np.mean(paired_margin)), "triplet_count": int(sum(counts)),
               "averaging": "equal_image_weight_then_all_source_supported_contradiction_triplets",
               "mean_worst_positive_margin": float(np.mean(np.asarray(low) - np.asarray(high))),
               "tie_rule": "strict_positive_over_every_contradiction", "neutral_as_negative": False}
    raw = {"image_ids": np.asarray(image_ids), "worst_positive_score": np.asarray(low),
           "best_negative_score": np.asarray(high), "joint_correct": np.asarray(joint),
           "paired_joint_accuracy": np.asarray(paired_accuracy), "paired_joint_margin": np.asarray(paired_margin),
           "triplet_count": np.asarray(counts), "raw_image_index": raw_i, "raw_text_index": raw_t,
           "raw_relation": raw_labels, "raw_score": scores}
    return summary, raw


def requested_epochs(specification, original_history, original_selection):
    available = {row["epoch"] for row in original_history["epochs"]}
    if specification == "all":
        requested = available
    elif specification == "selected":
        selected = original_selection.get("selected_epoch")
        if selected is None:
            raise ValueError("Original selection has no checkpoint; specify explicit epochs")
        requested = {selected}
    else:
        requested = {int(part) for part in specification.split(",")}
    requested.add(0)
    if not requested <= available or min(requested) < 0:
        raise ValueError(f"Requested unavailable epochs: {sorted(requested - available)}")
    return sorted(requested)


def verify_candidate(candidate):
    ledger_path = candidate / "ledger.json"
    ledger, history = read(ledger_path), read(candidate / "history.json")
    identity = ledger["identity"]
    if content_hash(identity) != ledger["ledger_sha256"] or history["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Candidate ledger identity is inconsistent")
    for filename, expected in identity["source_sha256"].items():
        verify_record({"path": filename, "sha256": expected})
    verify_record(identity["protocol"])
    return ledger, history


def load_checkpoint(candidate, row, ledger, initial):
    filename = (candidate / row["checkpoint"]["path"]).resolve()
    if not filename.is_relative_to(candidate) or sha256_file(filename) != row["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint path/hash mismatch")
    payload = torch.load(filename, map_location="cpu", weights_only=True)
    if payload["ledger_sha256"] != ledger["ledger_sha256"] or payload["epoch"] != row["epoch"]:
        raise ValueError("Checkpoint is not bound to candidate and epoch")
    if payload["seed"] != ledger["identity"]["config"]["seed"] or payload["model_config"] != ledger["identity"]["model"]:
        raise ValueError("Checkpoint model or seed differs from ledger")
    if payload["training_metadata"]["sha256"] != sha256_file(candidate / "ledger.json"):
        raise ValueError("Checkpoint training metadata binding differs")
    model = BoundedPairScorer(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if initial is None:
        if row["epoch"] != 0 or payload["optimizer_steps"] != 0 or payload["update_norm"] != 0:
            raise ValueError("Baseline is not the initial checkpoint")
        if torch.count_nonzero(model.output.weight) or torch.count_nonzero(model.output.bias):
            raise ValueError("Baseline residual head is nonzero")
    else:
        calculated = float(torch.sqrt(sum((parameter.detach() - initial[name]).double().square().sum()
                                          for name, parameter in model.named_parameters())))
        if not np.isclose(calculated, payload["update_norm"], atol=1e-12, rtol=1e-12):
            raise ValueError("Checkpoint update norm cannot be reproduced")
        if row["epoch"] > 0 and (payload["optimizer_steps"] <= 0 or calculated <= 0 or not torch.any(model.output.weight != 0)):
            raise ValueError("Trained checkpoint is zero or has no optimizer steps")
    for key in ("optimizer_steps", "update_norm"):
        if payload[key] != row[key]:
            raise ValueError(f"Checkpoint history mismatch: {key}")
    return model, payload, {"path": relative(filename), "sha256": sha256_file(filename)}


def run(args):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    amendment_path = verify_record({"path": args.amendment, "sha256": args.amendment_sha256})
    amendment = read(amendment_path)
    verify_record({"path": amendment["base_protocol"], "sha256": amendment["base_protocol_sha256"]})
    verify_record({"path": amendment["evaluator"], "sha256": amendment["evaluator_sha256"]})
    output = root_path(args.output)
    sources = {relative(__file__): sha256_file(Path(__file__)), amendment["evaluator"]: amendment["evaluator_sha256"],
               "src/gcr/practical_scorer.py": sha256_file(ROOT / "src/gcr/practical_scorer.py")}
    inputs_cache, shortlist_cache, results = {}, {}, []
    for candidate_value in args.candidate:
        candidate = root_path(candidate_value)
        if output == candidate or output.is_relative_to(candidate):
            raise ValueError("Canonical output must be separate from original candidate files")
        ledger, original_history = verify_candidate(candidate)
        identity = ledger["identity"]
        if identity["protocol"] != {"path": amendment["base_protocol"], "sha256": amendment["base_protocol_sha256"]}:
            raise ValueError("Amendment does not cover candidate's original protocol")
        input_key = content_hash(identity["inputs"])
        if input_key not in inputs_cache:
            inputs_cache[input_key] = load_inputs(identity)
        data, pool = inputs_cache[input_key]
        encoder, epsilon = identity["config"]["encoder"], float(identity["model"]["epsilon"])
        cache_key = (input_key, epsilon)
        if cache_key not in shortlist_cache:
            shortlist_cache[cache_key] = get_cached_shortlist(output, encoder, identity["inputs"], pool, epsilon, sources)
        cache, guard, cache_record = shortlist_cache[cache_key]
        target = output / encoder / candidate.name
        if target == candidate or target.is_relative_to(candidate) or candidate.is_relative_to(target):
            raise ValueError("Canonical candidate outputs must be separate from original candidate directories")
        binding = {"schema": "canonical_practical_development_v1", "candidate": relative(candidate),
                   "ledger": {"path": relative(candidate / "ledger.json"), "sha256": sha256_file(candidate / "ledger.json")},
                   "ledger_sha256": ledger["ledger_sha256"], "source_hashes": sources,
                   "amendment": {"path": relative(amendment_path), "sha256": args.amendment_sha256},
                   "protocol": identity["protocol"], "inputs": identity["inputs"], "shortlist": cache_record,
                   "torch_threads": 1, "canonical_inference": "one_pair_row_zero",
                   "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}
        immutable_json(target / "binding.json", binding)
        binding_record = {"path": relative(target / "binding.json"), "sha256": sha256_file(target / "binding.json")}
        original_selection = read(candidate / "selection.json")
        epochs = requested_epochs(args.epochs, original_history, original_selection)
        original_rows = {row["epoch"]: row for row in original_history["epochs"]}
        initial_model, _, _ = load_checkpoint(candidate, original_rows[0], ledger, None)
        initial = {name: parameter.detach().clone() for name, parameter in initial_model.named_parameters()}
        for epoch in epochs:
            record_path = target / f"epoch_{epoch:02d}.json"
            if record_path.exists():
                record = read(record_path)
                if record["binding"] != binding_record or record["checkpoint"]["sha256"] != original_rows[epoch]["checkpoint"]["sha256"]:
                    raise ValueError("Existing canonical epoch identity differs")
                verify_record(record["composition"]["predictions"])
                verify_record(record["retrieval"]["predictions"])
                continue
            model, payload, checkpoint_record = load_checkpoint(candidate, original_rows[epoch], ledger, initial if epoch else None)
            started = time.monotonic()
            effective_model = model if epoch else None
            composition, composition_raw = rescore_composition(effective_model, data)
            retrieval, retrieval_raw = rescore_retrieval(effective_model, pool, cache, guard)
            composition["predictions"] = save_npz(target / f"epoch_{epoch:02d}_composition.npz", composition_raw)
            retrieval["predictions"] = save_npz(target / f"epoch_{epoch:02d}_retrieval.npz", retrieval_raw)
            record = {"epoch": epoch, "checkpoint": checkpoint_record, "optimizer_steps": payload["optimizer_steps"],
                      "update_norm": payload["update_norm"], "binding": binding_record,
                      "composition": composition, "retrieval": retrieval, "elapsed_seconds": time.monotonic() - started}
            immutable_json(record_path, record)
            print(json.dumps({"event": "canonical_epoch_complete", "candidate": relative(candidate), "epoch": epoch,
                              "paired_joint_accuracy": composition["paired_joint_accuracy"], "i2t_r1": retrieval["i2t_r1"],
                              "t2i_r1": retrieval["t2i_r1"], "unique_scored_pairs": retrieval["unique_scored_pairs"],
                              "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
        rows = []
        for record_path in sorted(target.glob("epoch_[0-9][0-9].json")):
            record = read(record_path)
            if record["binding"] != binding_record:
                raise ValueError("Canonical history combines different inference bindings")
            verify_record(record["checkpoint"])
            verify_record(record["composition"]["predictions"])
            verify_record(record["retrieval"]["predictions"])
            rows.append(record)
        complete = [row["epoch"] for row in rows] == list(range(identity["config"]["epochs"] + 1))
        selection = select_development_epoch(rows)
        snapshot = {"schema": "canonical_practical_selection_snapshot_v1", "binding": binding_record, "epochs": rows,
                    "evaluated_epochs": [row["epoch"] for row in rows], "requested_epochs": epochs,
                    "trajectory_complete": complete, "selection_is_finalizable": complete,
                    "selection_scope": "complete_trajectory" if complete else "provisional_partial_trajectory",
                    "selection": selection, "original_selection": original_selection,
                    "original_history_snapshot": original_history, "practical_gate": "unchanged",
                    "held_out_evaluation": "not_performed", "current_test_outcomes_used_for_selection": False}
        snapshot_path = target / f"selection_{content_hash(snapshot)[:20]}.json"
        immutable_json(snapshot_path, snapshot)
        result = {"candidate": relative(candidate), "snapshot": relative(snapshot_path),
                  "snapshot_sha256": sha256_file(snapshot_path), "trajectory_complete": complete, "selection": selection}
        results.append(result)
        print(json.dumps({"event": "canonical_selection_snapshot", **result}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="Original candidate directory; repeatable")
    parser.add_argument("--epochs", default="selected", help="selected, all available, or a comma-separated epoch list; always includes 0")
    parser.add_argument("--output", required=True, help="Separate canonical development output root")
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "canonical_development_rescore_complete", "candidates": run(args)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
