#!/usr/bin/env python3
"""Canonical, locked evaluation of the calibrated source-pair v7 family.

No held-out token or prefix preparation starts until this distinct v7 study has
locked both encoders and all three seeds. One score is used for every pair and gallery.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_practical_v6 import (path, read, write, immutable, relative, digest, load_dataset,
                                  bootstrap, ENCODERS, SEEDS, DATASETS, FAMILY_SIZE, REPLICATES, BOOTSTRAP_SEED)
from gcr.practical_text_v6 import load_text_tower, tokenizer, WEIGHTS
from gcr.practical_text_training_v6 import LastTextBlock
from gcr.practical_text_inference_v6 import encode_token_rows
from gcr.practical_text_parallel_v6 import encode_parallel
from gcr.practical_text_parallel_v7 import encode_parallel as encode_parallel_capturing
FAMILY = "bounded_token_source_pair_calibrated_v7"
STUDY_SCHEMA = "sanw_practical_source_pair_token_v7"
SOURCE_FACTORY = "gcr.practical_source_pair_v7.SourcePairTrainingExamples"
EVALUATION_STATUS = "adaptive_exploratory_after_historical_and_v6_test_reuse"
ALPHA_GRID = (.1, .2, .35, .5, .75, 1.)
SOURCE_PATHS = ("scripts/evaluate_practical_text_v7.py", "scripts/audit_practical_text_v7.py",
                "src/gcr/practical_text_training_v7.py", "src/gcr/practical_source_pair_v7.py",
                "src/gcr/practical_development_v7.py", "scripts/calibrate_practical_text_development_v7.py",
                "scripts/audit_practical_text_inference_v6.py", "scripts/evaluate_practical_v6.py", "scripts/audit_practical_v6.py",
                "src/gcr/practical_text_v6.py", "src/gcr/practical_text_training_v6.py",
                "src/gcr/practical_text_inference_v6.py", "src/gcr/practical_text_parallel_v6.py",
                "src/gcr/practical_text_parallel_v7.py", "src/gcr/practical_prefix_capture_v7.py",
                "scripts/audit_practical_prefix_reuse_v7.py")


def _score_parameters(epsilon, alpha):
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Require positive epsilon and finite amplitude between zero and one")


def paired_scores(images, baseline_texts, delta, epsilon, alpha):
    _score_parameters(epsilon, alpha)
    image, baseline, difference = (np.asarray(v, dtype=np.float64) for v in (images, baseline_texts, delta))
    if image.ndim != 2 or image.shape != baseline.shape or image.shape != difference.shape:
        raise ValueError("Paired scoring shapes differ")
    base = np.sum(image * baseline, axis=1)
    if alpha == 0:
        return base
    return base + alpha * (epsilon * np.tanh(np.sum(image * difference, axis=1) / epsilon))


def score_matrix(images, baseline_texts, delta, epsilon, alpha):
    _score_parameters(epsilon, alpha)
    image, baseline, difference = (np.asarray(v, dtype=np.float64) for v in (images, baseline_texts, delta))
    if image.ndim != 2 or baseline.ndim != 2 or baseline.shape != difference.shape or image.shape[1] != baseline.shape[1]:
        raise ValueError("Matrix scoring shapes differ")
    base = image @ baseline.T
    if alpha == 0:
        return base
    return base + alpha * (epsilon * np.tanh((image @ difference.T) / epsilon))


def verified_record(record):
    if digest(path(record["path"])) != record["sha256"]:
        raise ValueError("Provenance record changed")
    return read(record["path"])


def validate_state_study(state, manifest, protocol_hash):
    """Prevent old-study checkpoints from being relabeled as new training."""
    if state.get("alpha") not in ALPHA_GRID:
        raise ValueError("Selected amplitude is outside the fixed positive grid")
    ledger = verified_record(state["training_ledger"])
    identity = ledger["identity"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != ledger["ledger_sha256"]:
        raise ValueError("Training ledger content hash differs")
    if identity.get("schema") != STUDY_SCHEMA or identity.get("source_factory") != SOURCE_FACTORY:
        raise ValueError("Checkpoint training ledger belongs to a different study")
    if identity["protocol"]["sha256"] != protocol_hash:
        raise ValueError("Training protocol differs from selected protocol")
    protocol = verified_record(identity["protocol"])
    if identity["source_sha256"] != protocol["source_hashes"]:
        raise ValueError("Training sources differ from pre-fit protocol")
    for filename, sha in identity["source_sha256"].items():
        if digest(path(filename)) != sha:
            raise ValueError("Frozen training source changed")
    for key, expected in protocol["fit_config"].items():
        if identity["config"].get(key) != expected:
            raise ValueError("Fit config differs from frozen protocol")
    if identity["config"]["encoder"] != manifest["encoder"] or identity["config"]["seed"] != state["seed"]:
        raise ValueError("Selected fit identity differs")
    payload = torch.load(path(state["checkpoint"]), map_location="cpu", weights_only=True)
    if payload.get("study_schema") != STUDY_SCHEMA or payload.get("source_factory") != SOURCE_FACTORY:
        raise ValueError("Checkpoint belongs to a different study")
    if payload["schema"] != "sanw_practical_text_last_block_v1" or payload["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Checkpoint architecture or training ledger differs")
    if payload["config"] != identity["config"] or payload["weight_identity"] != identity["weight_identity"]:
        raise ValueError("Checkpoint and ledger fitting identities differ")
    if payload["weight_identity"] != WEIGHTS[manifest["encoder"]] or float(payload["config"]["epsilon"]) != .01:
        raise ValueError("Wrong frozen encoder or trained residual budget")
    if payload["epoch"] != state["epoch"] or payload["optimizer_steps"] != state["optimizer_steps"] or payload["optimizer_steps"] <= 0:
        raise ValueError("Selected training progress differs")
    if not np.isfinite(payload["update_norm"]) or payload["update_norm"] <= 0 or payload["update_norm"] != state["update_norm"]:
        raise ValueError("A genuinely updated selected checkpoint is required")
    squares = []
    for name, tensor in payload["state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite fitted tensor")
        if name.startswith("block."):
            reference = "reference_" + name
        elif name.startswith("final_norm."):
            reference = name.replace("final_norm.", "reference_norm.", 1)
        else:
            continue
        squares.append((tensor - payload["state_dict"][reference]).double().square().sum())
    actual_update = float(torch.sqrt(sum(squares)))
    if actual_update <= 0 or abs(actual_update - state["update_norm"]) > 1e-10:
        raise ValueError("Fitted tensor update disagrees with selected state")
    return payload


def validate_development_selection(state, manifest, protocol_hash):
    snapshot = verified_record(state["canonical_development_snapshot"])
    if snapshot["schema"] != "canonical_source_pair_calibration_v7" or snapshot["encoder"] != manifest["encoder"] or snapshot["seed"] != state["seed"]:
        raise ValueError("Canonical development snapshot identity differs")
    if snapshot["trajectory_complete"] is not True or snapshot["selection_is_finalizable"] is not True or snapshot["current_test_outcomes_used_for_selection"] is not False or snapshot["held_out_evaluation"] != "not_performed":
        raise ValueError("Only complete development-only selection can be locked")
    binding = verified_record(snapshot["binding"])
    if binding["ledger"] != state["training_ledger"] or binding["protocol"]["sha256"] != protocol_hash:
        raise ValueError("Development snapshot was not bound to this fit and protocol")
    candidates = snapshot["candidates"]
    if {(r["epoch"], r["alpha"]) for r in candidates} != {(e, a) for e in range(1, 5) for a in ALPHA_GRID} or len(candidates) != 24:
        raise ValueError("Canonical development grid is incomplete")
    eligible = []
    for row in candidates:
        if row["update_norm"] <= 0 or row["optimizer_steps"] <= 0 or row["residual_rms"] <= 1e-10:
            continue
        if row["composition"]["paired_joint_accuracy"] <= row["frozen_composition"]["paired_joint_accuracy"]:
            continue
        passing = True
        for direction in ("i2t", "t2i"):
            effect = row["retention"][direction]
            if effect["replicates"] != REPLICATES or effect["family_size"] != FAMILY_SIZE or effect["bootstrap_seed"] != BOOTSTRAP_SEED:
                raise ValueError("Development uncertainty contract differs")
            passing &= effect["difference"] >= 0 and effect["ci_lower"] > -.01
        if passing:
            eligible.append(row)
    if not eligible:
        raise ValueError("No candidate passed the declared development gate")
    selected = max(eligible, key=lambda r: (r["composition"]["paired_joint_accuracy"] - r["frozen_composition"]["paired_joint_accuracy"], r["composition"]["mean_paired_joint_margin"], -r["epoch"], -r["alpha"]))
    selection = snapshot["selection"]
    if (state["epoch"], state["alpha"]) != (selected["epoch"], selected["alpha"]) or (selection["selected_epoch"], selection["selected_alpha"]) != (selected["epoch"], selected["alpha"]):
        raise ValueError("Selected epoch/amplitude disagrees with canonical development selector")
    if selected["checkpoint"] != {"path": state["checkpoint"], "sha256": state["checkpoint_sha256"]}:
        raise ValueError("Selected development checkpoint changed")
    return selected


def verify_manifest(filename, protocol_hash):
    m = read(filename)
    if m["encoder"] not in ENCODERS or m["protocol_sha256"] != protocol_hash or m["family"] != FAMILY:
        raise ValueError("Token family manifest identity mismatch")
    if m.get("test_outcomes_used_for_selection") is not False or m.get("current_test_outcomes_used_for_selection") is not False:
        raise ValueError("Current test outcomes cannot select states")
    if m.get("evaluation_status") != EVALUATION_STATUS or m.get("historical_and_v6_test_exposure") is not True:
        raise ValueError("Adaptive historical and v6 test exposure must be explicit")
    if m.get("study_schema") != STUDY_SCHEMA:
        raise ValueError("Wrong v7 study schema")
    if len(m["states"]) != 3 or sorted(s["seed"] for s in m["states"]) != list(SEEDS):
        raise ValueError("Exactly three selected seeds required")
    if len({s["state_id"] for s in m["states"]}) != 3:
        raise ValueError("Duplicate state IDs")
    for state in m["states"]:
        if Path(state["state_id"]).name != state["state_id"] or state["state_id"] in (".", "..", "frozen"):
            raise ValueError("Unsafe state identifier")
        if state["epoch"] < 1 or state["update_norm"] <= 0:
            raise ValueError("A nonzero fitted state is required")
        if digest(path(state["checkpoint"])) != state["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        validate_state_study(state, m, protocol_hash)
        validate_development_selection(state, m, protocol_hash)
    if not m.get("source_hashes"):
        raise ValueError("Source provenance is required")
    for filename, sha in m["source_hashes"].items():
        if digest(path(filename)) != sha:
            raise ValueError("Training or selection source changed")
    if digest(path(m["training_metadata"]["path"])) != m["training_metadata"]["sha256"]:
        raise ValueError("Training feature metadata changed")
    return m


def create_lock(protocol, protocol_hash, manifests, prior_v6_audit=None):
    if digest(path(protocol)) != protocol_hash:
        raise ValueError("Protocol digest differs")
    protocol_data = read(protocol)
    if protocol_data.get("family") != FAMILY or protocol_data.get("ownership", {}).get("checkpoint_study_schema") != STUDY_SCHEMA or protocol_data.get("source_factory") != SOURCE_FACTORY:
        raise ValueError("Wrong pre-fit v7 protocol schema")
    if protocol_data["practical_gate"] != verified_record(protocol_data["parent_protocols"][0])["practical_gate"]:
        raise ValueError("Inherited practical gate changed")
    if protocol_data["development"]["alpha_grid"] != list(ALPHA_GRID):
        raise ValueError("Predeclared amplitude grid changed")
    records = {}
    for filename in manifests:
        manifest = verify_manifest(filename, protocol_hash)
        if manifest["encoder"] in records:
            raise ValueError("Duplicate encoder")
        records[manifest["encoder"]] = {"manifest": relative(filename), "manifest_sha256": digest(path(filename))}
    if set(records) != set(ENCODERS):
        raise ValueError("Both encoders must be selected before scoring")
    audit_record = None
    if prior_v6_audit is not None:
        audit_record = prior_v6_audit if isinstance(prior_v6_audit, dict) else {"path": relative(prior_v6_audit), "sha256": digest(path(prior_v6_audit))}
        if verified_record(audit_record).get("passed") is not True:
            raise ValueError("Prefix reuse requires a successful prior independent v6 audit")
    return {"schema_version": 1, "family": FAMILY, "protocol": relative(protocol), "protocol_sha256": protocol_hash,
            "encoders": records, "source_hashes": {s: digest(ROOT / s) for s in SOURCE_PATHS},
            "family_size": FAMILY_SIZE, "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "evaluation_status": EVALUATION_STATUS,
            "inference": "Single-caption first11 prefix and learned/reference suffix, ownEOT trim, float32, Torchthreads1; float64 score s0+alpha*epsilon*tanh(v dot delta/epsilon); alpha fixed outside tanh for every task/pair/gallery.",
            "study_schema": STUDY_SCHEMA, "historical_and_v6_test_exposure": True,
            "prior_v6_independent_audit": audit_record}


def load_lock(filename, sha):
    if digest(path(filename)) != sha:
        raise ValueError("Token selection lock changed")
    lock = read(filename)
    if create_lock(lock["protocol"], lock["protocol_sha256"], [r["manifest"] for r in lock["encoders"].values()], lock.get("prior_v6_independent_audit")) != lock:
        raise ValueError("Locked token selection changed")
    return lock


def load_models(manifest, tower):
    models = []
    frozen_reference = LastTextBlock(tower).state_dict()
    for state in manifest["states"]:
        payload = validate_state_study(state, manifest, manifest["protocol_sha256"])
        config = payload["config"]
        if payload["schema"] != "sanw_practical_text_last_block_v1" or config["encoder"] != manifest["encoder"] or config["seed"] != state["seed"] or payload["epoch"] != state["epoch"]:
            raise ValueError("Token checkpoint header mismatch")
        if payload["optimizer_steps"] <= 0 or abs(payload["update_norm"] - state["update_norm"]) > 1e-12:
            raise ValueError("Token checkpoint update differs")
        if payload["weight_identity"] != WEIGHTS[manifest["encoder"]]:
            raise ValueError("Token checkpoint pretrained weights differ")
        for name, value in frozen_reference.items():
            if name.startswith("reference_") or name == "projection":
                if not torch.equal(value, payload["state_dict"][name]):
                    raise ValueError("Frozen reference/projection changed during fitting")
        model = LastTextBlock(tower)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        model.epsilon = float(config["epsilon"])
        models.append(model)
    return models


def retrieval(dataset, delta, epsilon, alpha, block_size=64):
    a, manifest = dataset["arrays"], dataset["manifest"]
    images, texts = a["image_features"], a["text_features"]
    ilook, tlook = ({str(v): i for i, v in enumerate(a[k])} for k in ("image_ids", "text_ids"))
    owners = np.empty(len(texts), dtype=np.int64)
    for pair in manifest["pairs"]:
        owners[tlook[str(pair["text_id"])]] = ilook[str(pair["image_id"])]
    image_top = np.empty(len(images), dtype=np.int64)
    image_score = np.empty(len(images), dtype=np.float64)
    text_top = np.zeros(len(texts), dtype=np.int64)
    text_score = np.full(len(texts), -np.inf, dtype=np.float64)
    max_correction = 0.
    for start in range(0, len(images), block_size):
        scores = score_matrix(images[start:start + block_size], texts, delta, epsilon, alpha)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite full-gallery scores")
        rows = np.arange(len(scores))
        winners = np.argmax(scores, axis=1)
        image_top[start:start + len(scores)], image_score[start:start + len(scores)] = winners, scores[rows, winners]
        twinners = np.argmax(scores, axis=0)
        tscores = scores[twinners, np.arange(len(texts))]
        improved = tscores > text_score
        text_score[improved], text_top[improved] = tscores[improved], twinners[improved] + start
        change = scores - images[start:start + block_size].astype(np.float64) @ texts.astype(np.float64).T
        max_correction = max(max_correction, float(np.max(np.abs(change))))
    i2t, t2i = owners[image_top] == np.arange(len(images)), text_top == owners
    raw = {"image_ids": a["image_ids"], "text_ids": a["text_ids"], "text_source_image_ids": a["image_ids"][owners],
           "i2t_top_indices": image_top, "i2t_top_scores": image_score, "i2t_correct": i2t,
           "t2i_top_indices": text_top, "t2i_top_scores": text_score, "t2i_correct": t2i}
    metrics = {"images": len(images), "texts": len(texts), "i2t": {"r1": float(i2t.mean())}, "t2i": {"r1": float(t2i.mean())},
               "epsilon": epsilon, "alpha": alpha, "maximum_score_correction": max_correction,
               "tie_policy": "descending shared float64 score matrix; ascending candidate manifest index"}
    return metrics, raw


def triplets(dataset, delta, epsilon, alpha):
    a, rows = dataset["arrays"], dataset["manifest"]["triplets"]
    ilook, tlook = ({str(v): i for i, v in enumerate(a[k])} for k in ("image_ids", "text_ids"))
    raw = {"item_ids": np.asarray([str(r["id"]) for r in rows]), "image_ids": np.asarray([str(r["image_id"]) for r in rows]),
           "categories": np.asarray([str(r["category"]) for r in rows])}
    images = a["image_features"][[ilook[i] for i in raw["image_ids"]]]
    for field in ("positive1", "positive2", "negative"):
        ids = np.asarray([str(r[f"{field}_id"]) for r in rows])
        indices = [tlook[i] for i in ids]
        raw[f"{field}_ids"] = ids
        raw[f"{field}_scores"] = paired_scores(images, a["text_features"][indices], delta[indices], epsilon, alpha)
    raw["positive1_correct"] = raw["positive1_scores"] > raw["negative_scores"]
    raw["positive2_correct"] = raw["positive2_scores"] > raw["negative_scores"]
    raw["correct"] = raw["positive1_correct"] & raw["positive2_correct"]
    metrics = {"items": len(rows), "images": len(set(raw["image_ids"])), "positive1_accuracy": float(raw["positive1_correct"].mean()),
               "positive2_accuracy": float(raw["positive2_correct"].mean()), "both_accuracy": float(raw["correct"].mean())}
    return metrics, raw


def evaluate(args):
    lock = load_lock(args.selection_lock, args.selection_lock_sha256)
    entry = lock["encoders"][args.encoder]
    manifest = read(entry["manifest"])
    config = read(args.dataset_config)
    if config["encoder"] != args.encoder:
        raise ValueError("Encoder dataset config differs")
    training_metadata = read(manifest["training_metadata"]["path"])
    datasets = {name: load_dataset(config, name, args.encoder, training_metadata) for name in DATASETS}
    tower = load_text_tower(args.encoder, ROOT / "data/practical_v6_models")
    models = load_models(manifest, tower)
    states = [{"state_id": "frozen", "seed": None, "epoch": 0, "update_norm": 0., "checkpoint": None}] + manifest["states"]
    receipt = {"schema_version": 1, "encoder": args.encoder, "family": FAMILY,
               "selection_lock": relative(args.selection_lock), "selection_lock_sha256": args.selection_lock_sha256,
               "manifest": entry["manifest"], "manifest_sha256": entry["manifest_sha256"],
               "dataset_config": relative(args.dataset_config), "dataset_config_sha256": digest(path(args.dataset_config)),
               "states": states, "input_hashes": {n: d["hashes"] for n, d in datasets.items()},
               "source_hashes": lock["source_hashes"], "weight_identity": WEIGHTS[args.encoder],
               "evaluation_status": EVALUATION_STATUS, "historical_and_v6_test_exposure": True, "study_schema": STUDY_SCHEMA, "torch_threads": 1,
               "block_size": args.block_size, "text_inference": lock["inference"], "encoding_workers": args.encoding_workers,
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}
    out = path(args.output)
    immutable(out / "prescore_receipt.json", receipt)
    receipt_sha = digest(out / "prescore_receipt.json")
    index = {"schema_version": 1, "status": "running", "encoder": args.encoder,
             "prescore_receipt": relative(out / "prescore_receipt.json"), "prescore_receipt_sha256": receipt_sha,
             "runs": [{"state": s, "datasets": {}} for s in states]}
    tokenize = tokenizer()
    for name, dataset in datasets.items():
        encoding_paths = [out / "text_encodings" / s["state_id"] / f"{name}.npz" for s in manifest["states"]]
        encoding_meta = [p.with_suffix(".json") for p in encoding_paths]
        tokens = tokenize([r["text"] for r in dataset["manifest"]["texts"]])
        token_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        encoded = None
        if any(p.exists() for p in encoding_paths + encoding_meta):
            if not all(p.exists() for p in encoding_paths + encoding_meta):
                raise ValueError("Incomplete token encoding set requires explicit recovery")
        else:
            def progress(value):
                print(json.dumps(dict(event="canonical_test_text_encoding", encoder=args.encoder, dataset=name, **value)), flush=True)
            encode = encode_parallel_capturing if lock["prior_v6_independent_audit"] is not None else encode_parallel
            encoded = encode(args.encoder, ROOT / "data/practical_v6_models",
                [path(s["checkpoint"]) for s in manifest["states"]], tokens.numpy(),
                out / "encoding_work" / name, workers=args.encoding_workers, progress=progress)
            for k, (archive, metadata, state) in enumerate(zip(encoding_paths, encoding_meta, manifest["states"], strict=True)):
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, text_ids=dataset["arrays"]["text_ids"], delta=encoded["delta"][k])
                write(metadata, {"state": state, "dataset": name, "prescore_receipt_sha256": receipt_sha,
                                 "archive_sha256": digest(archive), "tokens_sha256": token_sha,
                                 "canonical_prefix_sha256": encoded["canonical_prefix_sha256"],
                                 "canonical_prefix_chunks": encoded["canonical_prefix_chunks"],
                                 "prefix_cache_chunks": encoded.get("prefix_cache_chunks"),
                                 "prefix_hash_policy": encoded["prefix_hash_policy"],
                                 "unique_token_sequences": encoded["unique_token_sequences"],
                                 "reference_sha256": hashlib.sha256(encoded["reference"][k].tobytes()).hexdigest(),
                                 "learned_sha256": hashlib.sha256(encoded["learned"][k].tobytes()).hexdigest()})
        for k, state in enumerate(states):
            provenance = {"state": state, "dataset": name, "prescore_receipt_sha256": receipt_sha}
            if k == 0:
                delta = np.zeros_like(dataset["arrays"]["text_features"])
                epsilon, alpha = .01, 0.
                encoding_record = None
            else:
                archive, metadata = encoding_paths[k - 1], encoding_meta[k - 1]
                emeta = read(metadata)
                if emeta["archive_sha256"] != digest(archive) or emeta["tokens_sha256"] != token_sha or emeta["prescore_receipt_sha256"] != receipt_sha or emeta["state"] != state or emeta["dataset"] != name:
                    raise ValueError("Saved token encoding provenance differs")
                with np.load(archive, allow_pickle=False) as z:
                    if not np.array_equal(z["text_ids"], dataset["arrays"]["text_ids"]):
                        raise ValueError("Token encoding order differs")
                    delta = z["delta"]
                epsilon, alpha = models[k - 1].epsilon, float(state["alpha"])
                encoding_record = {"archive": relative(archive), "archive_sha256": digest(archive),
                                   "metadata": relative(metadata), "metadata_sha256": digest(metadata)}
            archive = out / "predictions" / state["state_id"] / f"{name}.npz"
            metadata = archive.with_suffix(".json")
            if archive.exists() or metadata.exists():
                saved = read(metadata)
                if saved["provenance"] != provenance or saved["predictions_sha256"] != digest(archive):
                    raise ValueError("Saved raw predictions changed")
            else:
                metrics, raw = triplets(dataset, delta, epsilon, alpha) if name == "sugarcrepe_pp" else retrieval(dataset, delta, epsilon, alpha, args.block_size)
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, **raw)
                saved = {"provenance": provenance, "metrics": metrics, "predictions_sha256": digest(archive)}
                write(metadata, saved)
            index["runs"][k]["datasets"][name] = {"predictions": relative(archive), "predictions_sha256": digest(archive),
                "metadata": relative(metadata), "metadata_sha256": digest(metadata), "metrics": saved["metrics"], "text_encoding": encoding_record}
            write(out / "index.json", index)
            print(json.dumps({"event": "token_dataset_scored", "encoder": args.encoder, "dataset": name, "state": state["state_id"]}), flush=True)
        del encoded
    index["status"] = "complete"
    write(out / "index.json", index)


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
              "evaluation_status": EVALUATION_STATUS, "historical_and_v6_test_exposure": True, "study_schema": STUDY_SCHEMA, "alpha": .05,
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
    p.add_argument("--prior-v6-audit", help="Optional successful independent v6 audit, bound before scoring for verified prefix reuse")
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--selection-lock", required=True)
    p.add_argument("--selection-lock-sha256", required=True)
    p.add_argument("--encoder", choices=ENCODERS, required=True)
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--encoding-workers", type=int, choices=range(1, 7), default=3)
    p = sub.add_parser("analyze")
    p.add_argument("--indices", nargs=2, required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "lock":
        immutable(args.output, create_lock(args.protocol, args.protocol_sha256, args.manifests, args.prior_v6_audit))
        print(json.dumps({"selection_lock": relative(args.output), "sha256": digest(path(args.output))}))
    elif args.command == "evaluate":
        evaluate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
