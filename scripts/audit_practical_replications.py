#!/usr/bin/env python3
"""Independently verify four completed pooled-scorer replication records.

Checks identities, inputs, source/checkpoint/archive digests, checkpoint state,
finite values, and development aggregates reconstructed from archived raw data.
This does not repeat model inference or read held-out benchmark inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repository.resolve()
    torch.set_num_threads(1)
    digests, inputs_verified, sources_verified, records = {}, set(), set(), []
    numeric_checks = 0

    def sha(path):
        path = path.resolve()
        if path not in digests:
            digests[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digests[path]

    def check_record(base, record):
        path = base / record["path"]
        assert path.is_file(), str(path)
        assert sha(path) == record["sha256"], str(path)
        return path

    def equal(left, right, name):
        nonlocal numeric_checks
        numeric_checks += 1
        assert math.isfinite(float(left)) and abs(float(left) - float(right)) <= 1e-12, (name, left, right)

    def finite_tree(value):
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            assert bool(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                finite_tree(child)
        elif isinstance(value, float):
            assert math.isfinite(value)

    manifest = json.loads((root / "data/visual_entailment/manifest.json").read_text())
    image_lookup = {row["id"]: i for i, row in enumerate(manifest["images"])}
    text_lookup = {row["id"]: i for i, row in enumerate(manifest["texts"])}
    relation_codes = {"source": 1, "supported": 2, "contradicted": 3, "neutral": 4}
    validation = {i for i, row in enumerate(manifest["images"]) if row["split"] == "validation"}
    relations = {(image_lookup[p["image_id"]], text_lookup[p["text_id"]]): relation_codes[p["relation"]]
                 for p in manifest["pairs"] if image_lookup[p["image_id"]] in validation}
    retrieval_manifest = json.loads((root / "data/review_followup/e_vil_dev900/manifest.json").read_text())
    retrieval_images = [row["id"] for row in retrieval_manifest["images"]]
    retrieval_texts = [row["id"] for row in retrieval_manifest["texts"]]
    ret_image_lookup = {value: i for i, value in enumerate(retrieval_images)}
    ret_text_lookup = {value: i for i, value in enumerate(retrieval_texts)}
    owner = np.full(len(retrieval_texts), -1, dtype=np.int64)
    for pair in retrieval_manifest["pairs"]:
        assert pair["relation"] == "source"
        owner[ret_text_lookup[pair["text_id"]]] = ret_image_lookup[pair["image_id"]]
    assert np.all(owner >= 0)

    for encoder in ("vit_b32", "rn50"):
        for seed in (29, 43):
            directory = root / "results/practical_v6" / encoder / f"eps_0.01_ret_0.25_seed_{seed}"
            ledger = json.loads((directory / "ledger.json").read_text())
            completion = json.loads((directory / "completion.json").read_text())
            history_document = json.loads((directory / "history.json").read_text())
            identity = ledger["identity"]
            identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            assert identity_hash == ledger["ledger_sha256"] == completion["ledger_sha256"] == history_document["ledger_sha256"]
            for item in identity["inputs"].values():
                inputs_verified.add(str(check_record(root, item).relative_to(root)))
            for name, expected in identity["source_sha256"].items():
                assert sha(root / name) == expected
                sources_verified.add(name)
            check_record(root, identity["protocol"])
            assert completion["status"] == "completed_development_only" and completion["epochs"] == 12
            assert completion["history_sha256"] == sha(directory / "history.json")
            assert completion["selection"] == json.loads((directory / "selection.json").read_text())
            history = history_document["epochs"]
            assert [row["epoch"] for row in history] == list(range(13))
            initial = None
            for row in history:
                checkpoint_path = check_record(directory, row["checkpoint"])
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                assert checkpoint["ledger_sha256"] == identity_hash
                assert checkpoint["epoch"] == row["epoch"] and checkpoint["seed"] == seed
                assert checkpoint["optimizer_steps"] == row["optimizer_steps"] == row["epoch"] * math.ceil(1200 / 64)
                assert checkpoint["model_config"] == identity["model"]
                assert checkpoint["config"] == identity["config"]
                assert checkpoint["training_metadata"]["sha256"] == sha(directory / "ledger.json")
                finite_tree(checkpoint)
                state = checkpoint["state_dict"]
                if initial is None:
                    initial = {name: value.clone() for name, value in state.items()}
                    assert torch.count_nonzero(state["output.weight"]) == 0 and torch.count_nonzero(state["output.bias"]) == 0
                assert torch.equal(state["image_mean"], initial["image_mean"]) and torch.equal(state["text_mean"], initial["text_mean"])
                update_norm = torch.sqrt(sum((value - initial[name]).double().square().sum()
                                             for name, value in state.items() if name not in ("image_mean", "text_mean"))).item()
                equal(update_norm, row["update_norm"], "update norm")
                equal(update_norm, checkpoint["update_norm"], "checkpoint update norm")
                if row["epoch"]:
                    assert update_norm > 0

                composition_path = check_record(directory, row["composition"]["predictions"])
                with np.load(composition_path, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                for value in arrays.values():
                    if value.dtype.kind in "fc":
                        assert np.isfinite(value).all()
                keys = list(zip(arrays["raw_image_index"].tolist(), arrays["raw_text_index"].tolist()))
                assert len(keys) == len(set(keys)) == len(relations)
                assert dict(zip(keys, arrays["raw_relation"].tolist())) == relations
                per_image_accuracy, per_image_margin, lows, highs = [], [], [], []
                all_correct, source_correct, support_correct, counts, included_ids = [], [], [], [], []
                for image_index in sorted(validation):
                    selected = arrays["raw_image_index"] == image_index
                    labels, scores = arrays["raw_relation"][selected], arrays["raw_score"][selected]
                    source, support, negative = [scores[labels == code] for code in (1, 2, 3)]
                    if min(len(source), len(support), len(negative)) == 0:
                        continue
                    margins = np.minimum(source[:, None, None], support[None, :, None]) - negative[None, None, :]
                    per_image_accuracy.append((margins > 0).mean()); per_image_margin.append(margins.mean())
                    lows.append(min(source.min(), support.min())); highs.append(negative.max())
                    all_correct.append(lows[-1] > highs[-1]); source_correct.append(source.min() > negative.max())
                    support_correct.append(support.min() > negative.max()); counts.append(margins.size)
                    included_ids.append(manifest["images"][image_index]["id"])
                assert arrays["image_ids"].astype(str).tolist() == included_ids
                for name, values in (("paired_joint_accuracy", per_image_accuracy), ("paired_joint_margin", per_image_margin),
                                     ("worst_positive_score", lows), ("best_negative_score", highs),
                                     ("triplet_count", counts), ("joint_correct", all_correct)):
                    assert np.allclose(arrays[name], values, atol=1e-12, rtol=0), name
                    numeric_checks += len(values)
                summaries = {"paired_joint_accuracy": np.mean(per_image_accuracy), "mean_paired_joint_margin": np.mean(per_image_margin),
                             "all_pairs_correct_count": sum(all_correct), "all_pairs_accuracy": np.mean(all_correct),
                             "source_all_correct_count": sum(source_correct), "supported_all_correct_count": sum(support_correct),
                             "triplet_count": sum(counts), "image_count": len(included_ids),
                             "mean_worst_positive_margin": np.mean(np.asarray(lows) - np.asarray(highs))}
                for name, value in summaries.items():
                    equal(row["composition"][name], value, name)

                retrieval_path = check_record(directory, row["retrieval"]["predictions"])
                with np.load(retrieval_path, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                for value in arrays.values():
                    if value.dtype.kind in "fc":
                        assert np.isfinite(value).all()
                assert arrays["image_ids"].astype(str).tolist() == retrieval_images
                assert arrays["text_ids"].astype(str).tolist() == retrieval_texts
                assert np.array_equal(arrays["owner"], owner)
                image_top, text_top = arrays["image_top_indices"], arrays["text_top_indices"]
                assert image_top.shape == (900, 1) and text_top.shape == (4500, 1)
                assert 0 <= image_top.min() <= image_top.max() < 4500
                assert 0 <= text_top.min() <= text_top.max() < 900
                image_correct = owner[image_top[:, 0]] == np.arange(900)
                text_correct = text_top[:, 0] == owner
                assert np.array_equal(image_correct, arrays["image_correct"])
                assert np.array_equal(text_correct, arrays["text_correct"])
                for name, value in {"i2t_r1": image_correct.mean(), "t2i_r1": text_correct.mean(),
                                    "image_correct_count": image_correct.sum(), "text_correct_count": text_correct.sum()}.items():
                    equal(row["retrieval"][name], value, name)
            records.append({"encoder": encoder, "seed": seed, "ledger_sha256": identity_hash,
                            "completion_sha256": sha(directory / "completion.json"), "checkpoints": 13, "prediction_archives": 26})
    output = args.output or root / "recovery/current_turn_audit/practical_v6_replication_integrity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "scope": "four_new_replicates_of_common_pooled_configuration",
              "run_count": 4, "checkpoint_count": 52, "prediction_archive_count": 104,
              "numeric_aggregate_checks": numeric_checks, "unique_files_hashed": len(digests),
              "input_paths_verified": sorted(inputs_verified), "source_paths_verified": sorted(sources_verified),
              "rows": records, "audit_source_sha256": sha(Path(__file__)),
              "model_inference_recomputed": False,
              "inference_note": "Separate canonical development rescoring is performed by the evaluation audit agent.",
              "held_out_benchmarks_read": False}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "path": str(output), "sha256": sha(output),
                      "numeric_checks": numeric_checks, "checkpoint_count": 52, "archives": 104}, indent=2))


if __name__ == "__main__":
    main()
