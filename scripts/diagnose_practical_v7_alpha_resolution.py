#!/usr/bin/env python3
"""One authorized development-only alpha-resolution diagnostic; no refitting.

This never changes official v7 selection or authorizes replication. All scores
reuse canonical development delta arrays and the frozen original feature caches.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_development_v7 import (
    calibrated_score, cluster_statistics, development_bootstrap, source_pair_validation_metrics,
)
from gcr.training import canonical_json
from calibrate_practical_text_development_v7 import raw_relation_pairs, retrieval_predictions
from rescore_practical_text_development_v6 import (
    immutable_json, load_development, path, read, record, save_npz, verify,
)


def histogram_key(delta, clusters):
    sums, counts = cluster_statistics(delta, clusters)
    if not np.all(counts == counts[0]) or not np.equal(sums, np.rint(sums)).all():
        raise ValueError("Diagnostic caching requires equal-size integer-valued cluster sums")
    values, frequencies = np.unique(sums.astype(np.int64), return_counts=True)
    identity = {"clusters": len(counts), "items_per_cluster": int(counts[0]),
                "sums": values.tolist(), "frequencies": frequencies.tolist(),
                "replicates": 100000, "bootstrap_seed": 20261007, "family_size": 80}
    return hashlib.sha256(canonical_json(identity)).hexdigest(), identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-record", required=True)
    parser.add_argument("--diagnostic-record-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    description_path = verify({"path": args.diagnostic_record, "sha256": args.diagnostic_record_sha256})
    description = read(description_path)
    if description["alphas"] != [i / 100 for i in range(10, 101)] or description["epochs"] != [1, 2, 3, 4]:
        raise ValueError("Diagnostic grid differs from the sole authorized refinement")
    for name, sha in description["source_hashes"].items():
        verify({"path": name, "sha256": sha})
    for name in ("parent_protocol", "official_stop_decision"):
        verify(description[name])
    output = path(description["output"])
    if output.exists():
        raise FileExistsError("Preserve existing diagnostic artifacts; no overwrite")
    output.mkdir(parents=True)
    cache, cache_hits, coarse_checks, all_rows, encoder_records = {}, 0, 0, [], []
    def bootstrap_cached(delta, clusters):
        nonlocal cache_hits
        key, identity = histogram_key(delta, clusters)
        if key not in cache:
            summary, samples = development_bootstrap(delta, clusters)
            sample_record = save_npz(output / "bootstrap_histograms" / f"{key}.npz", samples=samples)
            metadata = {"histogram": identity, "effect": summary, "samples": sample_record}
            filename = output / "bootstrap_histograms" / f"{key}.json"
            immutable_json(filename, metadata)
            cache[key] = (summary, record(filename))
        else:
            cache_hits += 1
        return cache[key]
    for pilot in description["pilots"]:
        encoder = pilot["encoder"]
        snapshot = read(verify(pilot["snapshot"]))
        binding = read(verify(snapshot["binding"]))
        ledger = read(verify(binding["ledger"]))
        for item in binding["inputs"].values():
            verify(item)
        data, pool, text_indices, keys, tokens = load_development(ledger["identity"])
        delta_record = snapshot["candidates"][0]["canonical_delta"]
        features = np.load(verify(delta_record), allow_pickle=False)
        if not np.array_equal(features["keys"], keys) or not np.array_equal(features["tokens"], tokens):
            raise ValueError("Canonical saved text IDs/tokens differ")
        deltas = features["delta"]
        if deltas.shape[0] != 5 or np.count_nonzero(deltas[0]):
            raise ValueError("Canonical development delta lacks unchanged baseline and four fitted epochs")
        images, texts = raw_relation_pairs(data)
        relation_images = data.images.numpy()[images].astype(np.float64)
        relation_base = np.sum(relation_images * data.texts.numpy()[texts].astype(np.float64), axis=1)
        positions = {index: position for position, index in enumerate(text_indices)}
        relation_positions = np.asarray([positions[index] for index in texts])
        retrieval_images = pool.images.numpy().astype(np.float64)
        retrieval_base = retrieval_images @ pool.texts.numpy().astype(np.float64).T
        frozen_comp, frozen_raw = source_pair_validation_metrics(data, images, texts, relation_base)
        frozen_ret = retrieval_predictions(retrieval_base, pool)
        if len(pool.images) != 900 or len(pool.texts) != 4500 or not np.all(np.bincount(pool.owner.numpy()) == 5):
            raise ValueError("Diagnostic requires the unchanged development gallery")
        original = {(row["epoch"], row["alpha"]): row for row in snapshot["candidates"]}
        encoder_rows, raw_records = [], []
        epsilon = ledger["identity"]["config"]["epsilon"]
        for epoch in description["epochs"]:
            state = original[epoch, .1]
            delta = deltas[epoch]
            relation_correction = epsilon * np.tanh(np.sum(relation_images * delta[relation_positions].astype(np.float64), axis=1) / epsilon)
            retrieval_correction = epsilon * np.tanh((retrieval_images @ delta[len(text_indices):].astype(np.float64).T) / epsilon)
            base_rms = max(float(np.sqrt(np.mean(relation_correction ** 2))), float(np.sqrt(np.mean(retrieval_correction ** 2))))
            saved = {key: [] for key in ("image_correct", "text_correct", "image_top_indices", "text_top_indices",
                                         "image_top_scores", "text_top_scores", "paired_joint_accuracy", "paired_joint_margin",
                                         "raw_relation_score", "raw_relation_residual")}
            epoch_rows = []
            for alpha in description["alphas"]:
                comp, comp_raw = source_pair_validation_metrics(data, images, texts, calibrated_score(relation_base, relation_correction, alpha))
                ret = retrieval_predictions(calibrated_score(retrieval_base, retrieval_correction, alpha), pool)
                i_effect, i_record = bootstrap_cached(ret["image_correct"].astype(int) - frozen_ret["image_correct"], np.arange(len(pool.images)))
                t_effect, t_record = bootstrap_cached(ret["text_correct"].astype(int) - frozen_ret["text_correct"], pool.owner.numpy())
                gain = comp["paired_joint_accuracy"] - frozen_comp["paired_joint_accuracy"]
                feasible = (gain > 0 and state["update_norm"] > 0 and state["optimizer_steps"] > 0 and alpha * base_rms > 1e-10
                            and i_effect["difference"] >= 0 and t_effect["difference"] >= 0
                            and i_effect["ci_lower"] > -.01 and t_effect["ci_lower"] > -.01)
                row = {"encoder": encoder, "epoch": epoch, "alpha": alpha, "checkpoint": state["checkpoint"],
                       "joint_gain": gain, "mean_joint_margin": comp["mean_paired_joint_margin"],
                       "composition": comp, "retention": {"i2t": i_effect, "t2i": t_effect},
                       "bootstrap_records": {"i2t": i_record, "t2i": t_record},
                       "diagnostic_feasible": feasible, "official_v7_selection": False}
                if (epoch, alpha) in original:
                    prior = original[epoch, alpha]
                    if comp != prior["composition"] or row["retention"] != prior["retention"]:
                        raise ValueError("Diagnostic failed exact reproduction of an original coarse-grid candidate")
                    coarse_checks += 1
                for key in ("image_correct", "text_correct", "image_top_indices", "text_top_indices", "image_top_scores", "text_top_scores"):
                    saved[key].append(ret[key])
                for key in ("paired_joint_accuracy", "paired_joint_margin"):
                    saved[key].append(comp_raw[key])
                saved["raw_relation_score"].append(comp_raw["raw_score"])
                saved["raw_relation_residual"].append(alpha * relation_correction)
                epoch_rows.append(row)
            raw_record = save_npz(output / encoder / f"epoch_{epoch:02d}_all_alpha_predictions.npz",
                alphas=np.asarray(description["alphas"]), image_ids=frozen_ret["image_ids"], text_ids=frozen_ret["text_ids"],
                owner=frozen_ret["owner"], composition_image_ids=frozen_raw["image_ids"],
                raw_image_index=images, raw_text_index=texts, triplet_count=frozen_raw["triplet_count"],
                frozen_image_correct=frozen_ret["image_correct"], frozen_text_correct=frozen_ret["text_correct"],
                frozen_paired_joint_accuracy=frozen_raw["paired_joint_accuracy"],
                **{key: np.stack(values) for key, values in saved.items()})
            for row_index, row in enumerate(epoch_rows):
                row["raw_predictions"] = {**raw_record, "alpha_row": row_index}
            encoder_rows.extend(epoch_rows); raw_records.append(raw_record)
            print(json.dumps({"event": "diagnostic_epoch_complete", "encoder": encoder, "epoch": epoch,
                              "alpha_count": len(epoch_rows), "feasible_count": sum(row["diagnostic_feasible"] for row in epoch_rows),
                              "unique_bootstrap_histograms": len(cache), "bootstrap_cache_hits": cache_hits}), flush=True)
        filename = output / encoder / "diagnostic_rows.json"
        immutable_json(filename, {"encoder": encoder, "rows": encoder_rows, "canonical_delta": delta_record,
                                  "pilot_snapshot": pilot["snapshot"], "raw_predictions": raw_records})
        encoder_records.append(record(filename)); all_rows.extend(encoder_rows); features.close()
    feasible = [row for row in all_rows if row["diagnostic_feasible"]]
    # Reverify the official stop artifact after the diagnostic computation.
    verify(description["official_stop_decision"])
    result = {"schema": "practical_v7_alpha_resolution_diagnostic_result_v1", "diagnostic_record": record(description_path),
              "official_stop_decision_unchanged": description["official_stop_decision"],
              "status": "diagnostic_feasible_intervals_found_new_protocol_required" if feasible else "no_0.01_spaced_feasible_alphas",
              "candidate_count": len(all_rows), "feasible_count": len(feasible), "feasible_candidates": feasible,
              "encoder_results": encoder_records, "coarse_grid_candidates_reproduced_exactly": coarse_checks,
              "unique_bootstrap_histograms": len(cache), "bootstrap_cache_hits": cache_hits,
              "new_fits": 0, "new_encoding": 0, "held_out_data_read": False,
              "replication_authorized": False, "further_refinement_authorized": False,
              "interpretation": "Exploratory development diagnostic only; no official v7 success or deployment claim."}
    immutable_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "feasible_count": len(feasible), "result": record(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
