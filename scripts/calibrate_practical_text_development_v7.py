#!/usr/bin/env python3
"""Canonical source-pair/alpha calibration on development data only.

Requires completed, separately authorized v7 fits and a pre-fit protocol.
Reuses verified v6 frozen prefix bytes; never reads a held-out dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_development_v7 import (
    ALPHA_GRID, calibrated_score, development_bootstrap,
    select_calibrated_development, source_pair_validation_metrics,
)
from gcr.practical_text_inference_v6 import encode_token_rows
from gcr.practical_text_v6 import load_text_tower
from rescore_practical_text_development_v6 import (
    CachedTower, candidate_ledger, immutable_json, load_development, load_states,
    path, read, record, rel, save_npz, verify,
)


def raw_relation_pairs(data):
    pairs = [(image, text) for image in data.split_indices["validation"] for text in sorted(data.pairs[image])]
    return tuple(np.asarray(values, dtype=np.int64) for values in zip(*pairs))


def validate_v7_ledger(ledger, protocol, protocol_record, encoder):
    identity = ledger["identity"]
    if identity["schema"] != "sanw_practical_source_pair_token_v7" or identity.get("protocol") != protocol_record:
        raise ValueError("Calibration requires a source-pair v7 fit bound to this exact pre-fit protocol")
    if identity.get("source_factory") != "gcr.practical_source_pair_v7.SourcePairTrainingExamples":
        raise ValueError("Candidate did not declare the required source-pair training factory")
    if identity["config"]["encoder"] != encoder or identity["config"]["seed"] not in (17, 29, 43):
        raise ValueError("Candidate encoder/seed differs from the declared v7 study")
    if not protocol.get("fit_config"):
        raise ValueError("Protocol does not declare the fixed fitting configuration")
    for name, expected in protocol["fit_config"].items():
        if identity["config"].get(name) != expected:
            raise ValueError(f"Candidate fitting configuration differs: {name}")
    if identity["source_sha256"] != protocol["source_hashes"]:
        raise ValueError("Candidate fitting sources differ from the pre-fit protocol")


def validate_v7_checkpoint_study(checkpoints, ledger):
    # The shared architecture loader validates every tensor and update already.
    # This extra read enforces v7 study ownership without modifying that frozen
    # loader. There is no architecture/schema inference from a filename.
    for checkpoint in checkpoints:
        payload = torch.load(verify(checkpoint), map_location="cpu", weights_only=True)
        if payload.get("study_schema") != "sanw_practical_source_pair_token_v7" or payload["ledger_sha256"] != ledger["ledger_sha256"]:
            raise ValueError("Checkpoint belongs to a different study or ledger")
        if payload.get("source_factory") != "gcr.practical_source_pair_v7.SourcePairTrainingExamples":
            raise ValueError("Checkpoint did not declare the source-pair training factory")


def retrieval_predictions(scores, pool):
    images, texts = scores.argmax(axis=1), scores.argmax(axis=0)
    owner = pool.owner.numpy()
    image_correct = owner[images] == np.arange(len(pool.images))
    text_correct = texts == owner
    return {"image_ids": np.asarray(pool.image_ids), "text_ids": np.asarray(pool.text_ids),
            "owner": owner, "image_correct": image_correct, "text_correct": text_correct,
            "image_top_indices": images[:, None], "text_top_indices": texts[:, None],
            "image_top_scores": scores[np.arange(len(images)), images][:, None],
            "text_top_scores": scores[texts, np.arange(len(texts))][:, None]}


def validate_cached_prefix(cache, identity, keys, tokens, tower):
    receipt = read(cache / "receipt.json")
    for item in receipt["files"].values():
        verify(item)
    if receipt["binding"]["inputs"] != identity["inputs"] or receipt["binding"]["weight_identity"] != identity["weight_identity"]:
        raise ValueError("Frozen prefix weights/inputs differ from v7 training")
    for name in ("src/gcr/practical_text_inference_v6.py", "src/gcr/practical_text_v6.py"):
        verify({"path": name, "sha256": receipt["binding"]["source_sha256"][name]})
    if not np.array_equal(np.load(cache / "tokens.npy", allow_pickle=False), tokens):
        raise ValueError("Canonical development tokens differ from frozen prefix cache")
    if not np.array_equal(np.load(cache / "keys.npy", allow_pickle=False), keys):
        raise ValueError("Canonical development caption IDs/order differ")
    values = np.load(cache / "prefix_values.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(cache / "offsets.npy", allow_pickle=False)
    return receipt, CachedTower(tower, tokens, values, offsets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--canonical-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    protocol_path = verify({"path": args.protocol, "sha256": args.protocol_sha256})
    protocol = read(protocol_path)
    if not protocol.get("source_hashes"):
        raise ValueError("Calibration requires the predeclared source-hash protocol")
    for name, expected in protocol["source_hashes"].items():
        verify({"path": name, "sha256": expected})
    if rel(__file__) not in protocol["source_hashes"] or "src/gcr/practical_development_v7.py" not in protocol["source_hashes"]:
        raise ValueError("Calibration implementation is absent from the pre-fit protocol")
    candidates = [path(value) for value in args.candidate]
    first = candidate_ledger(candidates[0]); identity = first["identity"]
    validate_v7_ledger(first, protocol, record(protocol_path), args.encoder)
    data, pool, text_indices, keys, tokens = load_development(identity)
    tower = load_text_tower(args.encoder, ROOT / "data/practical_v6_models")
    cache = path(args.canonical_prefix)
    prefix_receipt, cached_tower = validate_cached_prefix(cache, identity, keys, tokens, tower)
    images, texts = raw_relation_pairs(data)
    relation_images = data.images.numpy()[images].astype(np.float64)
    relation_texts = data.texts.numpy()[texts].astype(np.float64)
    relation_base = np.sum(relation_images * relation_texts, axis=1)
    positions = {index: position for position, index in enumerate(text_indices)}
    relation_positions = np.asarray([positions[index] for index in texts])
    retrieval_images = pool.images.numpy().astype(np.float64)
    retrieval_texts = pool.texts.numpy().astype(np.float64)
    retrieval_base = retrieval_images @ retrieval_texts.T
    frozen_comp, frozen_comp_raw = source_pair_validation_metrics(data, images, texts, relation_base)
    frozen_ret = retrieval_predictions(retrieval_base, pool)
    if len(pool.images) != 900 or len(pool.texts) != 4500 or not np.all(np.bincount(pool.owner.numpy()) == 5):
        raise ValueError("Predeclared dev900 requires exactly five source captions per image")
    output = path(args.output) / args.encoder
    if any(output == candidate or output.is_relative_to(candidate) for candidate in candidates):
        raise ValueError("Calibration outputs must be separate from training artifacts")
    snapshots, seen_seeds = [], set()
    for candidate in candidates:
        ledger = candidate_ledger(candidate)
        validate_v7_ledger(ledger, protocol, record(protocol_path), args.encoder)
        seed = ledger["identity"]["config"]["seed"]
        if seed in seen_seeds:
            raise ValueError("Duplicate seed in candidate calibration list")
        seen_seeds.add(seed)
        if ledger["identity"]["inputs"] != identity["inputs"] or ledger["identity"]["weight_identity"] != identity["weight_identity"]:
            raise ValueError("Candidate encoder/input identity mismatch")
        if ledger["identity"]["config"]["encoder"] != args.encoder:
            raise ValueError("Candidate encoder mismatch")
        models, payloads, checkpoints, original = load_states(candidate, ledger, tower)
        validate_v7_checkpoint_study(checkpoints, ledger)
        target = output / candidate.name
        if target.exists():
            raise FileExistsError("Preserve existing calibration output; do not overwrite")
        binding = {"schema": "canonical_source_pair_calibration_binding_v7", "candidate": rel(candidate),
                   "ledger": record(candidate / "ledger.json"), "ledger_sha256": ledger["ledger_sha256"],
                   "protocol": record(protocol_path), "inputs": identity["inputs"],
                   "source_sha256": protocol["source_hashes"], "canonical_prefix_receipt": record(cache / "receipt.json"),
                   "canonical_prefix_sha256": prefix_receipt["canonical_prefix_sha256"],
                   "score": "s0 + alpha * epsilon * tanh(dot(frozen_image,delta)/epsilon)",
                   "alpha_grid": ALPHA_GRID, "alpha_scope": "one scalar per selected checkpoint for every task, pair and gallery",
                   "held_out_evaluation": "not_performed"}
        immutable_json(target / "binding.json", binding)
        started = time.monotonic()
        def progress(event):
            print(json.dumps({"event": "v7_canonical_development_suffix", "encoder": args.encoder,
                              "candidate": candidate.name, **event, "elapsed_seconds": time.monotonic() - started}), flush=True)
        encoded = encode_token_rows(cached_tower, models, tokens, progress=progress)
        if encoded["canonical_prefix_sha256"] != prefix_receipt["canonical_prefix_sha256"] or np.count_nonzero(encoded["delta"][0]):
            raise ValueError("Canonical prefix identity or exact zero initialization failed")
        feature_record = save_npz(target / "canonical_delta.npz", keys=keys, tokens=tokens, delta=encoded["delta"])
        baseline = {"composition": frozen_comp, "composition_predictions": save_npz(target / "epoch_00_composition.npz", **frozen_comp_raw),
                    "retrieval_predictions": save_npz(target / "epoch_00_retrieval.npz", **frozen_ret), "checkpoint": checkpoints[0]}
        immutable_json(target / "epoch_00.json", baseline)
        history, epsilon = [], float(identity["config"]["epsilon"])
        for index in range(1, 5):
            payload, delta = payloads[index], encoded["delta"][index]
            relation_delta = delta[relation_positions].astype(np.float64)
            relation_correction = epsilon * np.tanh(np.sum(relation_images * relation_delta, axis=1) / epsilon)
            retrieval_delta = delta[len(text_indices):].astype(np.float64)
            retrieval_correction = epsilon * np.tanh((retrieval_images @ retrieval_delta.T) / epsilon)
            for alpha in ALPHA_GRID:
                tag = f"epoch_{payload['epoch']:02d}_alpha_{alpha:g}"
                comp, comp_raw = source_pair_validation_metrics(data, images, texts, calibrated_score(relation_base, relation_correction, alpha))
                comp_raw["raw_residual"] = alpha * relation_correction
                ret_raw = retrieval_predictions(calibrated_score(retrieval_base, retrieval_correction, alpha), pool)
                i_effect, i_draws = development_bootstrap(ret_raw["image_correct"].astype(int) - frozen_ret["image_correct"], np.arange(len(pool.images)))
                t_effect, t_draws = development_bootstrap(ret_raw["text_correct"].astype(int) - frozen_ret["text_correct"], pool.owner.numpy())
                rms = max(float(np.sqrt(np.mean((alpha * relation_correction) ** 2))),
                          float(np.sqrt(np.mean((alpha * retrieval_correction) ** 2))))
                row = {**payload, "alpha": alpha, "checkpoint": checkpoints[index], "composition": comp,
                       "frozen_composition": frozen_comp, "retention": {"i2t": i_effect, "t2i": t_effect},
                       "residual_rms": rms, "canonical_delta": feature_record,
                       "composition_predictions": save_npz(target / f"{tag}_composition.npz", **comp_raw),
                       "retrieval_predictions": save_npz(target / f"{tag}_retrieval.npz", **ret_raw),
                       "bootstrap_predictions": save_npz(target / f"{tag}_bootstrap.npz", i2t=i_draws, t2i=t_draws)}
                immutable_json(target / f"{tag}.json", row); history.append(row)
                print(json.dumps({"event": "v7_development_alpha_complete", "encoder": args.encoder,
                                  "candidate": candidate.name, "epoch": payload["epoch"], "alpha": alpha,
                                  "joint": comp["paired_joint_accuracy"], "i2t": i_effect, "t2i": t_effect}), flush=True)
        selection = select_calibrated_development(history)
        snapshot = {"schema": "canonical_source_pair_calibration_v7", "encoder": args.encoder,
                    "seed": seed,
                    "binding": record(target / "binding.json"), "baseline": baseline, "candidates": history,
                    "trajectory_complete": True, "selection_is_finalizable": True, "selection": selection,
                    "held_out_evaluation": "not_performed", "current_test_outcomes_used_for_selection": False}
        immutable_json(target / "snapshot.json", snapshot); snapshots.append(record(target / "snapshot.json"))
        print(json.dumps({"event": "v7_development_selection_complete", "encoder": args.encoder,
                          "candidate": candidate.name, "selection": selection, "snapshot": snapshots[-1]}), flush=True)
        del models, encoded, history
    immutable_json(output / "completion.json", {"schema": "canonical_source_pair_calibration_completion_v7",
                                                "encoder": args.encoder, "snapshots": snapshots,
                                                "held_out_evaluation": "not_performed"})


if __name__ == "__main__":
    main()
