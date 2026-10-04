#!/usr/bin/env python3
"""Canonical development-only token-family trajectories and selection snapshots.

Exact batch-one prefixes are cached through the frozen canonical helper before
all fits complete. Saved prefixes are reused through that same helper, without
altering arithmetic, token order, encoder weights, or original run artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcr.practical_text_inference_v6 import encode_token_rows, normalized_cache, paired_scores, score_matrix
from gcr.practical_text_training_v6 import LastTextBlock
from gcr.practical_text_v6 import WEIGHTS, load_text_tower, tokenizer
from gcr.practical_training import select_development_epoch
from gcr.review_training import SourceRetrievalPool
from gcr.training import FeatureDataset, canonical_json, sha256_file, atomic_json


HELPER = "src/gcr/practical_text_inference_v6.py"


def path(value):
    item = Path(value)
    item = (item if item.is_absolute() else ROOT / item).resolve()
    if not item.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {value}")
    return item


def rel(value):
    return str(path(value).relative_to(ROOT))


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read(value):
    return json.loads(path(value).read_text())


def record(value):
    filename = path(value)
    return {"path": rel(filename), "sha256": sha256_file(filename)}


def verify(item):
    filename = path(item["path"])
    if sha256_file(filename) != item["sha256"]:
        raise ValueError(f"Bound file changed: {filename}")
    return filename


def immutable_json(filename, value):
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.exists():
        if read(filename) != value:
            raise ValueError(f"Immutable record differs: {filename}")
        return
    with filename.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz(filename, **arrays):
    filename = path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return record(filename)


def candidate_ledger(candidate):
    ledger = read(candidate / "ledger.json")
    identity = ledger["identity"]
    if digest(identity) != ledger["ledger_sha256"]:
        raise ValueError("Invalid text training ledger identity")
    for name, expected in identity["source_sha256"].items():
        verify({"path": name, "sha256": expected})
    for item in identity["inputs"].values():
        verify(item)
    if identity["config"]["epochs"] != 4 or identity["config"]["epsilon"] != 0.01:
        raise ValueError("This canonical trajectory requires the declared four-epoch token family")
    return ledger


def load_development(identity):
    names = {name: verify(item) for name, item in identity["inputs"].items()}
    dimension = WEIGHTS[identity["config"]["encoder"]]["dimension"]
    data = FeatureDataset(ROOT, names["manifest"], names["features"], names["metadata"], dimension)
    pool = SourceRetrievalPool(ROOT, names["development_manifest"], names["development_features"], names["development_metadata"], data, dimension)
    for container in (data, pool):
        container.images = torch.from_numpy(normalized_cache(container.images.numpy()))
        container.texts = torch.from_numpy(normalized_cache(container.texts.numpy()))
    if len(data.split_indices["validation"]) != 100 or len(pool.images) != 900 or len(pool.texts) != 4500:
        raise ValueError("Unexpected development scope")
    dev_text_indices = sorted({j for i in data.split_indices["validation"] for j in data.pairs[i]})
    rows = [("visual_entailment:" + data.text_ids[j], data.manifest["texts"][j]["text"]) for j in dev_text_indices]
    rows += [("retrieval_development:" + str(row["id"]), row["text"]) for row in pool.manifest["texts"]]
    tokens = tokenizer()([text for _, text in rows]).numpy()
    keys = np.asarray([key for key, _ in rows])
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("Namespaced development text keys are not unique")
    prefix_receipt = read(identity["prefix_cache_receipt"]["path"])
    verify(identity["prefix_cache_receipt"])
    original_prefix = path(identity["prefix_cache_receipt"]["path"]).parent
    for name in ("keys.npy", "tokens.npy"):
        verify({"path": rel(original_prefix / name), "sha256": prefix_receipt["files"][name]["sha256"]})
    old_keys = np.load(original_prefix / "keys.npy", allow_pickle=False)
    old_tokens = np.load(original_prefix / "tokens.npy", mmap_mode="r", allow_pickle=False)
    lookup = {key: i for i, key in enumerate(old_keys.tolist())}
    if not np.array_equal(tokens, old_tokens[[lookup[key] for key in keys.tolist()]]):
        raise ValueError("Canonical development token IDs differ from training cache")
    return data, pool, dev_text_indices, keys, tokens


class RecordingTower:
    def __init__(self, tower, tokens, values, offsets):
        self.tower, self.tokens, self.values, self.offsets = tower, tokens, values, offsets
        self.text_projection = tower.text_projection
        self.first = {}
        for index, row in enumerate(tokens):
            self.first.setdefault(row.tobytes(), index)

    def eval(self):
        self.tower.eval()
        return self

    def prefix(self, tokens, trim=True):
        if tokens.shape != (1, 77) or not trim:
            raise ValueError("Canonical prefix must contain one caption and own-EOT trimming")
        key = tokens[0].numpy().tobytes()
        index = self.first[key]
        prefix, ends = self.tower.prefix(tokens, trim=True)
        lo, hi = self.offsets[index:index + 2]
        if prefix.shape != (1, hi - lo, 512) or int(ends[0]) != hi - lo - 1:
            raise ValueError("Canonical prefix length differs from bound token IDs")
        self.values[lo:hi] = prefix[0].numpy()
        return prefix, ends


class CachedTower:
    def __init__(self, tower, tokens, values, offsets):
        self.text_projection = tower.text_projection
        self.values, self.offsets = values, offsets
        self.first = {}
        for index, row in enumerate(tokens):
            self.first.setdefault(row.tobytes(), index)

    def eval(self):
        return self

    def prefix(self, tokens, trim=True):
        if tokens.shape != (1, 77) or not trim:
            raise ValueError("Cached canonical prefix requires one own-EOT caption")
        index = self.first[tokens[0].numpy().tobytes()]
        lo, hi = self.offsets[index:index + 2]
        values = np.array(self.values[lo:hi], dtype=np.float32, copy=True)
        return torch.from_numpy(values)[None], torch.tensor([hi - lo - 1], dtype=torch.long)


def prepare_prefix(output, tower, identity, keys, tokens, sources):
    cache = output / "canonical_prefix"
    distribution = importlib.metadata.distribution("open_clip_torch")
    tokenizer_records = {name: sha256_file(distribution.locate_file("open_clip/" + name))
                         for name in ("tokenizer.py", "bpe_simple_vocab_16e6.txt.gz")}
    binding = {"encoder": identity["config"]["encoder"], "weight_identity": identity["weight_identity"],
               "inputs": identity["inputs"], "training_prefix_receipt": identity["prefix_cache_receipt"],
               "source_sha256": sources, "tokenizer": {"version": distribution.version, "files": tokenizer_records},
               "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(), "keys": keys.tolist(),
               "canonical_policy": "one_caption_own_EOT_prefix_and_suffix_torch_threads_one"}
    if (cache / "receipt.json").exists():
        receipt = read(cache / "receipt.json")
        if receipt["binding"] != binding:
            raise ValueError("Existing canonical prefix has a different identity")
        for item in receipt["files"].values():
            verify(item)
        values = np.load(cache / "prefix_values.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(cache / "offsets.npy", allow_pickle=False)
        return cache, receipt, CachedTower(tower, tokens, values, offsets)
    if cache.exists():
        raise FileExistsError("Incomplete canonical prefix cache exists; preserve and inspect it")
    cache.mkdir(parents=True)
    offsets = np.r_[0, np.cumsum(tokens.argmax(axis=1) + 1)].astype(np.int64)
    np.save(cache / "tokens.npy", tokens, allow_pickle=False)
    np.save(cache / "keys.npy", keys, allow_pickle=False)
    np.save(cache / "offsets.npy", offsets, allow_pickle=False)
    values = np.lib.format.open_memmap(cache / "prefix_values.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 512))
    recorder = RecordingTower(tower, tokens, values, offsets)
    baseline = LastTextBlock(tower).eval()
    started = time.monotonic()
    def progress(event):
        print(json.dumps({"event": "canonical_development_prefix", "encoder": binding["encoder"],
                          **event, "elapsed_seconds": time.monotonic() - started}), flush=True)
    encoded = encode_token_rows(recorder, [baseline], tokens, progress=progress)
    if np.count_nonzero(encoded["delta"]) or not np.array_equal(encoded["learned"], encoded["reference"]):
        raise ValueError("Canonical initialization does not exactly reproduce zero residual")
    for index, row in enumerate(tokens):
        first = recorder.first[row.tobytes()]
        if first != index:
            values[offsets[index]:offsets[index + 1]] = values[offsets[first]:offsets[first + 1]]
    values.flush()
    del values
    np.save(cache / "initial_reference_features.npy", encoded["reference"][0], allow_pickle=False)
    receipt = {"schema": "canonical_token_development_prefix_v1", "binding": binding,
               "caption_count": len(tokens), "unique_token_sequences": encoded["unique_token_sequences"],
               "canonical_prefix_sha256": encoded["canonical_prefix_sha256"],
               "initial_residual_exactly_zero": True, "elapsed_seconds": time.monotonic() - started,
               "files": {item.name: record(item) for item in sorted(cache.glob("*.npy"))}}
    immutable_json(cache / "receipt.json", receipt)
    del baseline, encoded
    values = np.load(cache / "prefix_values.npy", mmap_mode="r", allow_pickle=False)
    return cache, receipt, CachedTower(tower, tokens, values, offsets)


def composition(data, dev_text_indices, delta, epsilon):
    positions = {text_index: position for position, text_index in enumerate(dev_text_indices)}
    image_indices, text_indices, labels, spans = [], [], [], []
    for image_index in data.split_indices["validation"]:
        lo = len(labels)
        for text_index in sorted(data.pairs[image_index]):
            image_indices.append(image_index); text_indices.append(text_index); labels.append(data.pairs[image_index][text_index])
        spans.append((image_index, lo, len(labels)))
    image_indices, text_indices, labels = map(np.asarray, (image_indices, text_indices, labels))
    images, texts = data.images.numpy()[image_indices], data.texts.numpy()[text_indices]
    differences = delta[[positions[index] for index in text_indices]]
    scores = paired_scores(images, texts, differences, epsilon)
    residuals = epsilon * np.tanh(np.sum(images.astype(np.float64) * differences.astype(np.float64), axis=1) / epsilon)
    ids, paired_acc, paired_margin, lows, highs, joint, source_all, support_all, counts = [], [], [], [], [], [], [], [], []
    for image_index, lo, hi in spans:
        local_labels, local_scores = labels[lo:hi], scores[lo:hi]
        source, supported, negative = [local_scores[local_labels == code] for code in (1, 2, 3)]
        if min(len(source), len(supported), len(negative)) == 0:
            continue
        margins = np.minimum(source[:, None, None], supported[None, :, None]) - negative[None, None, :]
        ids.append(data.image_ids[image_index]); paired_acc.append(float((margins > 0).mean()))
        paired_margin.append(float(margins.mean())); counts.append(margins.size)
        lows.append(float(min(source.min(), supported.min()))); highs.append(float(negative.max()))
        joint.append(lows[-1] > highs[-1]); source_all.append(source.min() > negative.max()); support_all.append(supported.min() > negative.max())
    summary = {"image_count": len(ids), "paired_joint_accuracy": float(np.mean(paired_acc)),
               "mean_paired_joint_margin": float(np.mean(paired_margin)), "triplet_count": int(sum(counts)),
               "all_pairs_correct_count": int(sum(joint)), "all_pairs_accuracy": float(np.mean(joint)),
               "source_all_correct_count": int(sum(source_all)), "supported_all_correct_count": int(sum(support_all)),
               "mean_worst_positive_margin": float(np.mean(np.asarray(lows) - highs)),
               "residual_rms": float(np.sqrt(np.mean(residuals ** 2))), "maximum_absolute_residual": float(np.abs(residuals).max()),
               "averaging": "equal_image_weight_then_all_source_supported_contradiction_triplets", "neutral_as_negative": False}
    raw = {"image_ids": np.asarray(ids), "paired_joint_accuracy": np.asarray(paired_acc), "paired_joint_margin": np.asarray(paired_margin),
           "triplet_count": np.asarray(counts), "worst_positive_score": np.asarray(lows), "best_negative_score": np.asarray(highs),
           "joint_correct": np.asarray(joint), "raw_image_index": image_indices, "raw_text_index": text_indices,
           "raw_relation": labels, "raw_score": scores, "raw_residual": residuals}
    return summary, raw


def retrieval(pool, delta, epsilon):
    images, texts = pool.images.numpy(), pool.texts.numpy()
    scores = score_matrix(images, texts, delta, epsilon)
    corrections = epsilon * np.tanh((images.astype(np.float64) @ delta.astype(np.float64).T) / epsilon)
    if not np.isfinite(scores).all() or not np.isfinite(corrections).all() or np.abs(corrections).max() > epsilon:
        raise ValueError("Invalid or unbounded canonical retrieval score")
    image_top, text_top = scores.argmax(axis=1), scores.argmax(axis=0)
    owner = pool.owner.numpy()
    image_correct, text_correct = owner[image_top] == np.arange(len(images)), text_top == owner
    summary = {"i2t_r1": float(image_correct.mean()), "t2i_r1": float(text_correct.mean()),
               "image_correct_count": int(image_correct.sum()), "text_correct_count": int(text_correct.sum()),
               "image_count": len(images), "text_count": len(texts), "relevance": "source_caption_ownership",
               "tie_rule": "descending_float64_score_then_ascending_gallery_manifest_index",
               "residual_rms": float(np.sqrt(np.mean(corrections ** 2))),
               "maximum_absolute_residual": float(np.abs(corrections).max())}
    raw = {"image_ids": np.asarray(pool.image_ids), "text_ids": np.asarray(pool.text_ids), "owner": owner,
           "image_correct": image_correct, "text_correct": text_correct,
           "image_top_indices": image_top[:, None], "text_top_indices": text_top[:, None],
           "image_top_scores": scores[np.arange(len(images)), image_top][:, None],
           "text_top_scores": scores[text_top, np.arange(len(texts))][:, None]}
    return summary, raw


def load_states(candidate, ledger, tower):
    original = read(candidate / "history.json")
    completion = read(candidate / "completion.json")
    if original["ledger_sha256"] != ledger["ledger_sha256"] or completion["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Run completion/history is not bound to its ledger")
    if completion["history_sha256"] != sha256_file(candidate / "history.json") or completion["status"] != "completed_development_only":
        raise ValueError("Run completion does not bind its final history")
    if [row["epoch"] for row in original["epochs"]] != list(range(5)):
        raise ValueError("Canonical token selection requires the complete epoch0–4 trajectory")
    models, payloads, checkpoint_records, initial = [], [], [], None
    for row in original["epochs"]:
        filename = (candidate / row["checkpoint"]["path"]).resolve()
        if not filename.is_relative_to(candidate) or sha256_file(filename) != row["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint path/hash mismatch")
        payload = torch.load(filename, map_location="cpu", weights_only=True)
        if payload["schema"] != "sanw_practical_text_last_block_v1" or payload["ledger_sha256"] != ledger["ledger_sha256"]:
            raise ValueError("Unexpected checkpoint schema or ledger")
        if payload["config"] != ledger["identity"]["config"] or payload["weight_identity"] != ledger["identity"]["weight_identity"]:
            raise ValueError("Checkpoint configuration differs from ledger")
        model = LastTextBlock(tower).eval()
        reference = {name: value.clone() for name, value in model.state_dict().items() if name.startswith("reference_") or name == "projection"}
        model.load_state_dict(payload["state_dict"], strict=True)
        for name, value in reference.items():
            if not torch.equal(value, model.state_dict()[name]):
                raise ValueError("Frozen reference/projection differs from the pinned pretrained model")
        if initial is None:
            initial = {name: parameter.detach().clone() for name, parameter in model.learned_named_parameters()}
        norm = float(torch.sqrt(sum((parameter.detach() - initial[name]).double().square().sum()
                                    for name, parameter in model.learned_named_parameters())))
        if payload["epoch"] != row["epoch"] or payload["optimizer_steps"] != row["optimizer_steps"]:
            raise ValueError("Checkpoint epoch/step mismatch")
        if not np.isclose(norm, row["update_norm"], atol=1e-12, rtol=1e-12) or not np.isclose(norm, payload["update_norm"], atol=1e-12, rtol=1e-12):
            raise ValueError("Checkpoint update norm cannot be reproduced")
        if row["epoch"] == 0 and (norm != 0 or payload["optimizer_steps"] != 0):
            raise ValueError("Nonzero epoch-zero state")
        if row["epoch"] and (norm <= 0 or payload["optimizer_steps"] <= 0):
            raise ValueError("Untrained nonzero epoch")
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite model state")
        models.append(model); payloads.append({key: payload[key] for key in ("epoch", "optimizer_steps", "update_norm")})
        checkpoint_records.append(record(filename))
        del payload, reference
    return models, payloads, checkpoint_records, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=WEIGHTS, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--wait-seconds", type=int, default=14400)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify({"path": HELPER, "sha256": args.helper_sha256})
    candidates = [path(value) for value in args.candidate]
    first_ledger = candidate_ledger(candidates[0])
    identity = first_ledger["identity"]
    if identity["config"]["encoder"] != args.encoder:
        raise ValueError("Candidate encoder mismatch")
    output = path(args.output) / args.encoder
    if any(output == candidate or output.is_relative_to(candidate) for candidate in candidates):
        raise ValueError("Canonical outputs must be separate from original run directories")
    source_names = (rel(__file__), HELPER, "src/gcr/practical_text_v6.py", "src/gcr/practical_text_training_v6.py",
                    "src/gcr/practical_training.py", "src/gcr/training.py", "src/gcr/review_training.py")
    sources = {name: sha256_file(ROOT / name) for name in source_names}
    data, pool, dev_text_indices, keys, tokens = load_development(identity)
    tower = load_text_tower(args.encoder, ROOT / "data/practical_v6_models")
    cache, cache_receipt, cached_tower = prepare_prefix(output, tower, identity, keys, tokens, sources)
    if args.prepare_only:
        print(json.dumps({"event": "canonical_development_prefix_ready", "encoder": args.encoder, "receipt": record(cache / "receipt.json")}), flush=True)
        return
    snapshots = []
    for candidate in candidates:
        deadline = time.monotonic() + args.wait_seconds
        while not (candidate / "completion.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for completed token trajectory: {candidate}")
            time.sleep(3)
        ledger = candidate_ledger(candidate)
        if ledger["identity"]["inputs"] != identity["inputs"] or ledger["identity"]["weight_identity"] != identity["weight_identity"]:
            raise ValueError("Candidate input or encoder weights differ from the development cache")
        for name, expected in sources.items():
            verify({"path": name, "sha256": expected})
        target = output / candidate.name
        if (target / "snapshot.json").exists():
            raise FileExistsError("Canonical snapshot already exists; do not overwrite a completed trajectory")
        models, payloads, checkpoint_records, original = load_states(candidate, ledger, tower)
        binding = {"schema": "canonical_token_development_binding_v1", "candidate": rel(candidate),
                   "ledger": record(candidate / "ledger.json"), "ledger_sha256": ledger["ledger_sha256"],
                   "inputs": identity["inputs"], "source_sha256": sources, "weight_identity": identity["weight_identity"],
                   "canonical_prefix_receipt": record(cache / "receipt.json"), "canonical_prefix_sha256": cache_receipt["canonical_prefix_sha256"],
                   "torch_threads": 1, "inference": "one_caption_own_EOT_then_float64_bounded_pair_score",
                   "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}
        immutable_json(target / "binding.json", binding)
        started = time.monotonic()
        def progress(event):
            print(json.dumps({"event": "canonical_development_suffix", "encoder": args.encoder,
                              "candidate": candidate.name, **event, "elapsed_seconds": time.monotonic() - started}), flush=True)
        encoded = encode_token_rows(cached_tower, models, tokens, progress=progress)
        if encoded["canonical_prefix_sha256"] != cache_receipt["canonical_prefix_sha256"]:
            raise ValueError("Reused canonical prefix bytes do not reproduce the original helper digest")
        if np.count_nonzero(encoded["delta"][0]):
            raise ValueError("Canonical epoch zero is not exactly unchanged")
        learned_record = save_npz(target / "canonical_text_features.npz", keys=keys, tokens=tokens,
                                 delta=encoded["delta"], learned=encoded["learned"], reference=encoded["reference"])
        history, epsilon = [], float(ledger["identity"]["config"]["epsilon"])
        for index, payload in enumerate(payloads):
            epoch, delta = payload["epoch"], encoded["delta"][index]
            comp, comp_raw = composition(data, dev_text_indices, delta[:len(dev_text_indices)], epsilon)
            ret, ret_raw = retrieval(pool, delta[len(dev_text_indices):], epsilon)
            if epoch and max(comp["residual_rms"], ret["residual_rms"]) <= 1e-10:
                raise ValueError("Nonzero epoch has effectively unchanged canonical scores")
            comp["predictions"] = save_npz(target / f"epoch_{epoch:02d}_composition.npz", **comp_raw)
            ret["predictions"] = save_npz(target / f"epoch_{epoch:02d}_retrieval.npz", **ret_raw)
            row = {**payload, "checkpoint": checkpoint_records[index], "composition": comp, "retrieval": ret,
                   "binding": record(target / "binding.json"), "canonical_text_features": learned_record,
                   "canonical_delta_rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))}
            immutable_json(target / f"epoch_{epoch:02d}.json", row)
            history.append(row)
            print(json.dumps({"event": "canonical_token_epoch_complete", "encoder": args.encoder,
                              "candidate": candidate.name, "epoch": epoch, "joint": comp["paired_joint_accuracy"],
                              "i2t": ret["i2t_r1"], "t2i": ret["t2i_r1"], "residual_rms": ret["residual_rms"]}), flush=True)
        selection = select_development_epoch(history)
        snapshot = {"schema": "canonical_token_development_selection_v1", "binding": record(target / "binding.json"),
                    "candidate": rel(candidate), "encoder": args.encoder, "seed": ledger["identity"]["config"]["seed"],
                    "epochs": history, "trajectory_complete": True, "selection_is_finalizable": True,
                    "selection": selection, "original_selection": read(candidate / "selection.json"),
                    "held_out_evaluation": "not_performed", "current_test_outcomes_used_for_selection": False}
        immutable_json(target / "snapshot.json", snapshot)
        snapshots.append(record(target / "snapshot.json"))
        print(json.dumps({"event": "canonical_token_selection_complete", "encoder": args.encoder,
                          "candidate": candidate.name, "selection": selection, "snapshot": snapshots[-1]}), flush=True)
        del models, encoded, history
    immutable_json(output / "completion.json", {"schema": "canonical_token_development_completion_v1", "encoder": args.encoder,
                                                "canonical_prefix_receipt": record(cache / "receipt.json"),
                                                "snapshots": snapshots, "held_out_evaluation": "not_performed"})
    print(json.dumps({"event": "canonical_token_development_complete", "encoder": args.encoder,
                      "completion": record(output / "completion.json")}), flush=True)


if __name__ == "__main__":
    main()
