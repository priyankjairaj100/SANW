#!/usr/bin/env python3
"""Independently reconstruct v8 pilot predictions, uncertainty, and stop rule.

This reads training/development inputs only. It cannot launch fitting or tests.
The production scorer, retrieval routine, metrics, bootstrap, and gate functions
are not used to reconstruct the numerical evidence.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from evaluate_practical_constrained_development_v8 import (
    digest, load_development, read, record, root_path, verified_run,
    verify_record, write_json,
)


def load_archive(entry):
    with np.load(verify_record(entry), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def representations(images, texts, model):
    image = np.einsum("nd,dr->nr", images - model.image_mean, model.image_basis, optimize=False)
    text = np.einsum("nd,dr->nr", texts - model.text_mean, model.text_basis, optimize=False)
    left = np.einsum("nd,dr->nr", image, model.coefficient, optimize=False)
    return left, text


def pair_values(images, texts, left, right, ii, jj):
    return np.sum(images[ii] * texts[jj], axis=1) + np.sum(left[ii] * right[jj], axis=1)


def check_retrieval(raw, images, texts, expected_owner, model):
    if not np.array_equal(raw["owner"], expected_owner):
        raise ValueError("Saved retrieval ownership differs from development manifest")
    left, right = (representations(images, texts, model) if model is not None
                   else (np.zeros((len(images), 1)), np.zeros((len(texts), 1))))
    # Full gallery, different global screening bound and block layout from the
    # producer. Every possible competitor against each claimed winner is scored
    # canonically, including all exact ties. No fixed top-k is used.
    approximate = images @ texts.T + left @ right.T
    guard = (128 * (images.shape[1] + left.shape[1] + 1) * np.finfo(np.float64).eps
             * (1 + np.linalg.norm(images, axis=1).max() * np.linalg.norm(texts, axis=1).max()
                + np.linalg.norm(left, axis=1).max() * np.linalg.norm(right, axis=1).max()))
    verified_pairs = 0
    for direction, scores in (("i2t", approximate), ("t2i", approximate.T)):
        for query, row in enumerate(scores):
            claimed_score = raw[f"{direction}_top_scores"][query]
            candidates = np.flatnonzero(row >= claimed_score - guard)
            if not len(candidates):
                raise ValueError("Claimed winner exceeds the exhaustive score bound")
            query_indices = np.full(len(candidates), query, dtype=np.int64)
            ii, jj = (query_indices, candidates) if direction == "i2t" else (candidates, query_indices)
            values = pair_values(images, texts, left, right, ii, jj)
            winner = int(np.lexsort((candidates, -values))[0])
            if candidates[winner] != raw[f"{direction}_top_indices"][query] or values[winner] != claimed_score:
                raise ValueError("Full-gallery canonical top-1 reconstruction disagrees")
            verified_pairs += len(values)
        winners = raw[f"{direction}_top_indices"]
        correct = expected_owner[winners] == np.arange(len(images)) if direction == "i2t" else winners == expected_owner
        if not np.array_equal(correct, raw[f"{direction}_correct"]):
            raise ValueError("Saved retrieval correctness disagrees with ownership")
        query, gallery = raw[f"{direction}_rescored_query_indices"], raw[f"{direction}_rescored_gallery_indices"]
        ii, jj = (query, gallery) if direction == "i2t" else (gallery, query)
        values = pair_values(images, texts, left, right, ii, jj)
        if not np.array_equal(values, raw[f"{direction}_rescored_scores"]):
            raise ValueError("Saved producer candidate scores changed")
    return {"full_gallery_pairs_screened": int(approximate.size), "independent_canonical_comparisons": int(verified_pairs)}


def check_composition(raw, data, model):
    expected = [(image, text, data.pairs[image][text]) for image in data.split_indices["validation"]
                for text in sorted(data.pairs[image])]
    ii, jj, labels = map(np.asarray, zip(*expected))
    for key, value in (("raw_image_index", ii), ("raw_text_index", jj), ("raw_relation", labels)):
        if not np.array_equal(value, raw[key]):
            raise ValueError("Composition relation units differ from the manifest")
    left, right = (representations(data.images, data.texts, model) if model is not None
                   else (np.zeros((len(data.images), 1)), np.zeros((len(data.texts), 1))))
    scores = pair_values(data.images, data.texts, left, right, ii, jj)
    if not np.array_equal(scores, raw["raw_score"]):
        raise ValueError("Canonical composition pair score reconstruction disagrees")
    key = lambda text: tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
    arrays = {name: [] for name in ("original", "source_pair")}
    ids = {name: [] for name in arrays}
    counts = {name: [] for name in arrays}
    for image in data.split_indices["validation"]:
        mask = ii == image
        lookup = dict(zip(jj[mask].tolist(), scores[mask].tolist()))
        codes = data.pairs[image]
        source, supported, negative = ([j for j in sorted(codes) if codes[j] == code] for code in (1, 2, 3))
        original = [min(lookup[s], lookup[p]) > lookup[n] for s in source for p in supported for n in negative]
        keys = {j: key(data.manifest["texts"][j]["text"]) for j in codes}
        positive_keys = {keys[j] for j in source + supported}
        source_pair = [min(lookup[a], lookup[b]) > lookup[n] for a, b in combinations(source, 2)
                       if keys[a] != keys[b] for n in negative if keys[n] not in positive_keys]
        for name, values in (("original", original), ("source_pair", source_pair)):
            if values:
                ids[name].append(data.image_ids[image]); counts[name].append(len(values)); arrays[name].append(np.mean(values))
    for name in arrays:
        for suffix, expected_values in (("image_ids", ids[name]), ("triplet_counts", counts[name]), ("joint_accuracy", arrays[name])):
            if not np.array_equal(np.asarray(expected_values), raw[f"{name}_{suffix}"]):
                raise ValueError("Composition image-level reconstruction disagrees")
    return {"pair_scores_reconstructed": len(scores), "original_triplets": sum(counts["original"]),
            "source_pair_triplets": sum(counts["source_pair"])}


def check_bootstrap(delta, ids, saved, summary):
    names = np.unique(ids)
    totals = np.asarray([np.sum(delta[ids == name]) for name in names])
    counts = np.asarray([np.sum(ids == name) for name in names])
    if not np.all(counts == counts[0]):
        raise ValueError("Pilot development scope unexpectedly has uneven image clusters")
    values = np.unique(totals)
    frequencies = np.asarray([np.sum(totals == value) for value in values])
    rng = np.random.default_rng(20261007)
    expected = np.empty(100000, dtype=np.float64)
    for start in range(0, 100000, 4096):
        draws = rng.multinomial(len(totals), frequencies / len(totals), size=min(4096, 100000 - start))
        expected[start:start + len(draws)] = np.dot(draws, values) / (len(totals) * counts[0])
    if not np.array_equal(expected, saved):
        raise ValueError("Stored bootstrap values differ from independent reconstruction")
    lower, upper = np.quantile(expected, [.05 / 160, 1 - .05 / 160], method="linear")
    required = {"difference": float(np.mean(delta)), "ci_lower": float(lower), "ci_upper": float(upper),
                "replicates": 100000, "bootstrap_seed": 20261007, "family_size": 80,
                "familywise_alpha": .05, "tail_probability": .05 / 160, "image_clusters": len(names), "items": len(delta)}
    if any(summary[name] != value for name, value in required.items()):
        raise ValueError("Bootstrap summary differs from independent reconstruction")
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--results", type=Path, nargs=2, required=True)
    parser.add_argument("--pilot-save", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-decision", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    if digest(root_path(args.protocol)) != args.protocol_sha256:
        raise ValueError("Protocol hash mismatch")
    protocol = read(args.protocol)
    pilot_save = read(args.pilot_save)
    archive = ROOT.parent / "practical_v8_checkpoints" / pilot_save["file"]
    if pilot_save["status"] != "saved_before_development_evaluation" or digest(archive) != pilot_save["sha256"]:
        raise ValueError("Pilot archive is not the preserved pre-development artifact")
    audit_rows = {}
    for filename in args.results:
        result = read(filename)
        evaluation_start = read(verify_record(result["start_receipt"]))
        completion_path = verify_record(evaluation_start["run"])
        for name in ("ledger", "checkpoint", "protocol"):
            verify_record(evaluation_start[name])
        identity, completion, run_protocol, model, norm = verified_run(completion_path.parent, args.protocol_sha256)
        encoder = identity["encoder"]
        if run_protocol != protocol or identity["config"]["seed"] != 17 or encoder in audit_rows:
            raise ValueError("Expected one protocol-matched seed-17 pilot per encoder")
        if result["encoder"] != encoder or result["seed"] != 17 or result["selected_epoch"] != completion["selected_epoch"]:
            raise ValueError("Result state does not match the verified training selection")
        data, pool = load_development(identity, protocol)
        raw = {name: load_archive(entry) for name, entry in result["artifacts"].items()}
        score_audits = {}
        for name, state in (("frozen", None), ("trained", model)):
            comp, ret = raw[f"{name}_composition"], raw[f"{name}_retrieval"]
            score_audits[name] = {"composition": check_composition(comp, data, state),
                                  "retrieval": check_retrieval(ret, pool.images, pool.texts, pool.owner.numpy(), state)}
        effects = {}
        f, t = raw["frozen_retrieval"], raw["trained_retrieval"]
        for direction, ids in (("i2t", f["image_ids"]), ("t2i", f["image_ids"][f["owner"]])):
            delta = t[f"{direction}_correct"].astype(np.float64) - f[f"{direction}_correct"].astype(np.float64)
            effects[direction] = check_bootstrap(delta, ids, raw["bootstrap_samples"][direction], result["retention_effects"][direction])
        f, t = raw["frozen_composition"], raw["trained_composition"]
        gains = {}
        for kind in ("original", "source_pair"):
            delta = t[f"{kind}_joint_accuracy"] - f[f"{kind}_joint_accuracy"]
            effects[kind] = check_bootstrap(delta, f[f"{kind}_image_ids"], raw["bootstrap_samples"][kind], result["composition_effects"][kind])
            gains[kind] = float(t[f"{kind}_joint_accuracy"].mean() - f[f"{kind}_joint_accuracy"].mean())
        checks = {"nonzero_trained_update": norm > 0, "original_joint_improvement": gains["original"] > 0,
                  "source_pair_joint_non_decrease": gains["source_pair"] >= 0,
                  "i2t_mean_non_decrease": effects["i2t"]["difference"] >= 0,
                  "i2t_strict_adjusted_retention": effects["i2t"]["ci_lower"] > -.01,
                  "t2i_mean_non_decrease": effects["t2i"]["difference"] >= 0,
                  "t2i_strict_adjusted_retention": effects["t2i"]["ci_lower"] > -.01}
        if checks != result["gate"]["checks"] or all(checks.values()) != result["gate"]["passed"]:
            raise ValueError("Development gate reconstruction disagrees")
        audit_rows[encoder] = {"result": record(filename), "start_receipt": result["start_receipt"],
                               "selected_epoch": completion["selected_epoch"], "gate": result["gate"],
                               "effects_reconstructed": effects, "score_audits": score_audits,
                               "bootstrap_values_bitwise_reconstructed": 400000}
    if set(audit_rows) != {"vit_b32", "rn50"}:
        raise ValueError("Both encoder pilots required")
    allowed = all(row["gate"]["passed"] for row in audit_rows.values())
    if allowed:
        raise ValueError("Both pilots passed; this failure-stop writer must not be used")
    audit = {"schema": "sanw_v8_independent_development_audit_v1", "passed": True,
             "passed_means": "evidence_reconstruction_not_scientific_gate", "protocol": record(args.protocol),
             "source": record(__file__), "encoders": audit_rows, "pilot_save": record(args.pilot_save),
             "preserved_pilot_archive": {"filename": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size},
             "bootstrap_values_reconstructed": 800000, "no_heldout_benchmark_access": True,
             "elapsed_seconds": time.monotonic() - start}
    audit_record = write_json(args.output, audit)
    decision = {"schema": "sanw_v8_pilot_stop_decision_v1", "protocol": record(args.protocol),
                "pilot_results": {encoder: row["result"] for encoder, row in audit_rows.items()},
                "audit": audit_record, "pilot_persistence_receipt": record(args.pilot_save),
                "both_pilot_gates_passed": allowed, "replication_authorized": False,
                "heldout_evaluation_authorized": False, "seeds_29_43_fit": False,
                "stop_reason": "the protocol requires both encoder pilots to pass; both failed",
                "status": "v8_stopped_after_development; practical_goal_unsolved",
                "no_checkpoint_or_alpha_search": True, "no_test_outcomes_used": True}
    write_json(args.stop_decision, decision)
    print(json.dumps({"audit": audit_record, "stop_decision": record(args.stop_decision),
                      "both_pilot_gates_passed": allowed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
