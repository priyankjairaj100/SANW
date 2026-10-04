#!/usr/bin/env python3
"""Recompute complete-gallery retention certificates without production diagnostics.

The audit recomputes every query from the feature cache and checkpoint weights.
It enumerates positive-set pooling sizes instead of using the production pool
loop. It keeps stable benchmark ranks separate from pessimistic certificates.
Coverage for nonzero updates is reported separately from unchanged states.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform

import numpy as np
import torch
import torch.nn.functional as F

from audit_allocation_distillation_independent import Audit, canonical_digest, digest, read_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "5624342a20dd33b4f235264f57599a8bad0486d70fa587a6d9269371ea00b5d4"
ENCODERS = ("vit_b32", "rn50")
FAMILIES = ("source", "supported", "allocation", "distilled", "wise_ft", "image_source_only", "reverse_source_only")
DATASETS = ("e_vil_test1000", "coco_karpathy")
MASKS = ("certified", "logit_drift_certified", "pairwise_certified", "combined_certified")
BOOLS = MASKS + ("teacher_correct", "student_correct")
FLOATS = ("threshold", "divergence", "probability_margin", "logit_margin",
          "logit_drift_oscillation", "logit_drift_linf_centered", "pairwise_threshold")
TEMPERATURE, ATOL, RTOL = 2., 1e-12, 1e-10
DIAGNOSTIC_SOURCES = {"scripts/diagnose_strengthen_retention.py", "scripts/evaluate_strengthen_retention.py",
    "src/gcr/rank_retention.py", "scripts/evaluate_allocation_distillation.py", "scripts/evaluate_study.py",
    "scripts/evaluate_review_followup.py", "src/gcr/adapters.py", "src/gcr/evaluation.py",
    "src/gcr/allocation_distillation_analysis.py"}


def r1_threshold(probabilities, relevant):
    """Enumerate feasible active positive sets for the closest R1 failure."""
    q = np.asarray(probabilities, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any(q < 0) or q.sum() <= 0:
        raise ValueError("Invalid teacher distribution")
    q = q / q.sum()
    relevant = np.unique(np.asarray(relevant, dtype=np.int64))
    if not len(relevant) or relevant[0] < 0 or relevant[-1] >= len(q):
        raise ValueError("Invalid relevance indices")
    wrong = np.ones(len(q), dtype=bool)
    wrong[relevant] = False
    if not wrong.any():
        return float("inf")
    b = float(q[wrong].max())
    a = np.sort(q[relevant])[::-1]
    if a[0] <= b:
        return 0.
    sums = np.cumsum(a)
    for count in range(1, len(a) + 1):
        cutoff = float((sums[count - 1] + b) / (count + 1))
        next_value = a[count] if count < len(a) else -np.inf
        if a[count - 1] > cutoff and next_value <= cutoff:
            pooled = np.append(a[:count], b)
            positive = pooled > 0
            return max(0., float(np.sum(pooled[positive] * np.log(pooled[positive] / cutoff))))
    raise ValueError("No feasible positive-set projection")


def strict_pass(value, threshold):
    return bool(np.isposinf(threshold) or value + ATOL + RTOL * abs(threshold) < threshold)


def independent_rows(teacher_scores, student_scores, relevance, logit_scale):
    """Evaluate native logits, forward KL, positive-set cost, and stable ranks."""
    teacher_scores = np.asarray(teacher_scores, dtype=np.float64)
    student_scores = np.asarray(student_scores, dtype=np.float64)
    if (teacher_scores.ndim != 2 or teacher_scores.shape != student_scores.shape or
            len(relevance) != len(teacher_scores) or not teacher_scores.shape[1] or
            not np.isfinite(teacher_scores).all() or not np.isfinite(student_scores).all() or
            not np.isfinite(logit_scale) or logit_scale <= 0):
        raise ValueError("Invalid full-gallery scores")
    n = len(teacher_scores)
    result = {key: np.zeros(n, dtype=bool) for key in BOOLS}
    result.update({key: np.zeros(n, dtype=np.float64) for key in FLOATS})
    result.update({key: np.zeros(n, dtype=np.int64) for key in
                   ("benchmark_teacher_ranks", "benchmark_student_ranks")})
    indices = np.arange(teacher_scores.shape[1])
    for row, (ts, ss) in enumerate(zip(teacher_scores, student_scores, strict=True)):
        relevant = np.unique(np.asarray(relevance[row], dtype=np.int64))
        if not len(relevant) or relevant[0] < 0 or relevant[-1] >= len(ts):
            raise ValueError("Invalid relevance indices")
        wrong = np.ones(len(ts), dtype=bool)
        wrong[relevant] = False
        z, v = ts * logit_scale, ss * logit_scale
        zshift, vshift = (z - z.max()) / TEMPERATURE, (v - v.max()) / TEMPERATURE
        logq = zshift - np.log(np.exp(zshift).sum())
        logp = vshift - np.log(np.exp(vshift).sum())
        q = np.exp(logq)
        divergence = max(0., float(np.dot(q, logq - logp)))
        threshold = r1_threshold(q, relevant)
        if wrong.any():
            teacher_correct = bool(z[relevant].max() > z[wrong].max())
            student_correct = bool(v[relevant].max() > v[wrong].max())
            a, b = float(q[relevant].max()), float(q[wrong].max())
            probability_margin = a - b
            logit_margin = float(z[relevant].max() - z[wrong].max())
            pooled = (a + b) / 2
            pairwise = (a * np.log(a / pooled) + (b * np.log(b / pooled) if b else 0.)) if a > b else 0.
        else:
            teacher_correct = student_correct = True
            probability_margin = logit_margin = pairwise = float("inf")
        drift = v - z
        oscillation = float(np.ptp(drift))
        values = {"threshold": threshold, "divergence": divergence,
                  "probability_margin": probability_margin, "logit_margin": logit_margin,
                  "logit_drift_oscillation": oscillation, "logit_drift_linf_centered": oscillation / 2,
                  "pairwise_threshold": pairwise, "teacher_correct": teacher_correct,
                  "student_correct": student_correct, "certified": strict_pass(divergence, threshold),
                  "logit_drift_certified": strict_pass(oscillation, logit_margin),
                  "pairwise_certified": strict_pass(divergence, pairwise)}
        values["combined_certified"] = values["certified"] or values["logit_drift_certified"]
        for key in MASKS:
            if values[key] and not (teacher_correct and student_correct):
                raise ValueError(f"Independent {key} contradicts actual rank")
        for key, value in values.items():
            result[key][row] = value
        for label, scores in (("teacher", ts), ("student", ss)):
            best = relevant[np.argmax(scores[relevant])]
            result[f"benchmark_{label}_ranks"][row] = (1 + np.count_nonzero(scores > scores[best]) +
                np.count_nonzero((scores == scores[best]) & (indices < best)))
    return result


def forward_features(features, weights=None):
    """Reconstruct only the declared float32 linear residual forward operation."""
    outputs = []
    with torch.inference_mode():
        for kind in ("image", "text"):
            values, batches = features[f"{kind}_features"], []
            for start in range(0, len(values), 1024):
                x = torch.from_numpy(np.ascontiguousarray(values[start:start + 1024], dtype=np.float32))
                if weights is not None:
                    x = x + F.linear(x, weights[f"{kind}.weight"])
                batches.append(F.normalize(x, p=2., dim=-1).cpu().numpy())
            outputs.append(np.concatenate(batches).astype(np.float64))
    return outputs


def compare_arrays(actual, expected, maxima):
    if set(actual) != set(expected):
        raise ValueError("Diagnostic array inventory differs")
    for key, a in actual.items():
        a, b = np.asarray(a), np.asarray(expected[key])
        if a.shape != b.shape:
            raise ValueError(f"Diagnostic array shape differs: {key}")
        if key in BOOLS or "ranks" in key:
            if a.dtype != b.dtype or not np.array_equal(a, b):
                raise ValueError(f"Diagnostic discrete array differs: {key}")
        else:
            if np.isnan(a).any() or np.isnan(b).any() or not np.array_equal(np.isinf(a), np.isinf(b)):
                raise ValueError(f"Diagnostic numeric validity differs: {key}")
            finite = np.isfinite(a)
            difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
            maxima[key] = max(maxima[key], difference)
            if not np.allclose(a, b, rtol=1e-11, atol=2e-12):
                raise ValueError(f"Independent diagnostic numeric mismatch: {key}: {difference}")


def independent_summary(raw):
    teacher, student = raw["teacher_correct"], raw["student_correct"]
    retained = teacher & student
    counts = {"queries": len(teacher), "teacher_correct": int(teacher.sum()),
              "student_correct": int(student.sum()), "actually_retained": int(retained.sum()),
              "benchmark_teacher_correct": int((raw["benchmark_teacher_ranks"] <= 1).sum()),
              "benchmark_student_correct": int((raw["benchmark_student_ranks"] <= 1).sum())}
    coverage = {}
    for key in MASKS:
        mask = raw[key]
        if np.any(mask & ~retained):
            raise ValueError("Saved certificate contradicts actual diagnostic correctness")
        certified = int(mask.sum())
        coverage[key] = {"certified_queries": certified, "false_certificates": 0,
            "fraction_of_all_queries": certified / len(teacher),
            "fraction_of_teacher_correct": certified / counts["teacher_correct"] if counts["teacher_correct"] else None,
            "fraction_of_actually_retained": certified / counts["actually_retained"] if counts["actually_retained"] else None,
            "teacher_correct_denominator": counts["teacher_correct"],
            "actually_retained_denominator": counts["actually_retained"]}
    numerics = {}
    for key in ("threshold", "divergence", "logit_margin", "logit_drift_oscillation",
                "logit_drift_linf_centered", "pairwise_threshold"):
        values = raw[key][np.isfinite(raw[key])]
        numerics[key] = {"finite_count": len(values), "infinite_count": int(np.isinf(raw[key]).sum()),
                        "mean": float(values.mean()) if len(values) else None,
                        "maximum": float(values.max()) if len(values) else None,
                        "median": float(np.median(values)) if len(values) else None}
    return {"counts": counts, "certificate_coverage": coverage, "numerics": numerics,
            "tie_disagreements": {"teacher_stable_vs_pessimistic": int(np.count_nonzero((raw["benchmark_teacher_ranks"] <= 1) != teacher)),
                                  "student_stable_vs_pessimistic": int(np.count_nonzero((raw["benchmark_student_ranks"] <= 1) != student))}}


def checked_archive(audit, record):
    audit.check_hash(record["predictions"], record["predictions_sha256"])
    with np.load(audit.path(record["predictions"]), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def expected_states(manifest):
    chosen = [row for row in manifest["selections"] if row["family"] in FAMILIES and row["tolerance_pp"] in (0., 1.)]
    for tolerance in (0., 1.):
        rows = [row for row in chosen if row["tolerance_pp"] == tolerance]
        if len(rows) != 21 or {(row["family"], row["seed"]) for row in rows} != {(family, seed) for family in FAMILIES for seed in (17, 29, 43)}:
            raise ValueError("Incomplete certificate selection roles")
    states = {row["state_id"]: row for row in manifest["states"]}
    if len(states) != len(manifest["states"]):
        raise ValueError("Duplicate training state IDs")
    if any(row["state_id"] not in states or states[row["state_id"]]["seed"] != row["seed"] for row in chosen):
        raise ValueError("Certificate selection seed differs from its state")
    return [{**states[sid], "selection_roles": [{key: row[key] for key in ("family", "tolerance_pp", "seed")}
             for row in chosen if row["state_id"] == sid]} for sid in sorted({row["state_id"] for row in chosen})]


def load_dataset(audit, config, name, input_hashes):
    paths = config["datasets"][name]
    for kind, path in paths.items():
        audit.check_hash(path, input_hashes[f"{kind}_sha256"])
    manifest = read_json(audit.path(paths["manifest"]))
    with np.load(audit.path(paths["features"]), allow_pickle=False) as archive:
        features = {key: archive[key] for key in archive.files}
    all_images = [str(row["id"]) for row in manifest["images"]]
    text_ids = [str(row["id"]) for row in manifest["texts"]]
    if list(features["image_ids"]) != all_images or list(features["text_ids"]) != text_ids:
        raise ValueError("Feature identities differ from gallery manifest")
    dimension = 512 if config["encoder"] == "vit_b32" else 1024
    for kind in ("image", "text"):
        values = features[f"{kind}_features"]
        if (values.shape != (len(features[f"{kind}_ids"]), dimension) or not np.isfinite(values).all() or
                not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=1e-4)):
            raise ValueError("Malformed or non-unit feature vectors")
    mask = np.asarray([row["split"] == "test" for row in manifest["images"]])
    features["image_ids"] = features["image_ids"][mask]
    features["image_features"] = features["image_features"][mask]
    image_ids = list(map(str, features["image_ids"]))
    expected_images = 1000 if name == "e_vil_test1000" else 5000
    if len(image_ids) != expected_images or len(text_ids) != 5 * expected_images:
        raise ValueError("Certificate audit requires the complete gallery")
    images, texts = {key: index for index, key in enumerate(image_ids)}, {key: index for index, key in enumerate(text_ids)}
    if len(images) != len(image_ids) or len(texts) != len(text_ids):
        raise ValueError("Duplicate gallery identities")
    i2t, t2i = [[] for _ in image_ids], [[] for _ in text_ids]
    for row in manifest["pairs"]:
        if row["relation"] != "source":
            raise ValueError("Certificates require source ownership relevance")
        i, j = images[str(row["image_id"])], texts[str(row["text_id"])]
        i2t[i].append(j)
        t2i[j].append(i)
    if any(len(row) != 5 or len(set(row)) != 5 for row in i2t) or any(len(row) != 1 for row in t2i):
        raise ValueError("Incomplete or duplicate source ownership")
    identity = {"image_ids": features["image_ids"], "text_ids": features["text_ids"],
                "text_source_image_ids": np.asarray([image_ids[row[0]] for row in t2i], str)}
    return features, (i2t, t2i), identity


def audit_index(audit, index_path, config_path, lock, lock_hash, block_size, maxima):
    index = read_json(audit.path(index_path))
    encoder = index["encoder"]
    if encoder not in ENCODERS or index.get("status") != "complete" or index["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("A complete retention certificate index is required")
    audit.check_hash(index["prediagnostic_receipt"], index["prediagnostic_receipt_sha256"])
    receipt = read_json(audit.path(index["prediagnostic_receipt"]))
    for key in ("protocol_sha256", "encoder", "source_hashes", "temperature", "native_logit_scale", "k", "scope", "evidence_type"):
        if index[key] != receipt[key]:
            raise ValueError(f"Diagnostic index differs from receipt: {key}")
    if set(receipt["source_hashes"]) != DIAGNOSTIC_SOURCES:
        raise ValueError("Diagnostic source hash inventory differs")
    if (receipt["temperature"] != TEMPERATURE or receipt["k"] != 1 or receipt["atol"] != ATOL or
            receipt["rtol"] != RTOL or receipt["selection_modified"] or receipt["success_gates_modified"] or
            receipt["selection_lock_sha256"] != lock_hash):
        raise ValueError("Certificate contract or selection lock differs")
    for name, expected in receipt["source_hashes"].items():
        audit.check_hash(name, expected)
    locked = lock["encoders"][encoder]
    audit.check_hash(locked["manifest"], locked["manifest_sha256"])
    manifest_path = audit.path(locked["manifest"])
    manifest = read_json(manifest_path)
    if (receipt["manifest_sha256"] != locked["manifest_sha256"] or not manifest["matched_controls_complete"] or
            manifest["test_outcomes_used_for_selection"] or manifest["selection_sha256"] != locked["selection_sha256"] or
            manifest["protocol_sha256"] != PROTOCOL_SHA256 or manifest["encoder"] != encoder):
        raise ValueError("Diagnostic selections differ from the shared lock")
    for label, expected in locked["selection_sha256"].items():
        audit.check_hash(manifest_path.parent / f"selection_{label}.json", expected)
    ledger = read_json(manifest_path.parent / "ledger.json")
    if (canonical_digest(ledger["identity"]) != ledger["ledger_sha256"] or
            ledger["ledger_sha256"] != manifest["ledger_sha256"] or ledger["ledger_sha256"] != locked["ledger_sha256"]):
        raise ValueError("Training ledger identity differs")
    for name, expected in ledger["identity"]["source_sha256"].items():
        audit.check_hash(name, expected)
    if ledger["identity"]["inputs"]["logit_scale"] != receipt["native_logit_scale"]:
        raise ValueError("Native logit scale differs from training")
    audit.check_hash(config_path, receipt["dataset_config_sha256"])
    config = read_json(audit.path(config_path))
    if config["encoder"] != encoder:
        raise ValueError("Dataset config encoder differs")
    audit.check_hash(receipt["evaluation_index"], receipt["evaluation_index_sha256"])
    evaluation = read_json(audit.path(receipt["evaluation_index"]))
    if (evaluation.get("status") != "complete" or evaluation["protocol_sha256"] != PROTOCOL_SHA256 or
            evaluation["selection_lock_sha256"] != lock_hash or evaluation["manifest_sha256"] != locked["manifest_sha256"] or
            evaluation["encoder"] != encoder):
        raise ValueError("Certificate audit requires the locked complete evaluation")
    audit.check_hash(evaluation["prescore_receipt"], evaluation["prescore_receipt_sha256"])
    prescore = read_json(audit.path(evaluation["prescore_receipt"]))
    for key in ("protocol_sha256", "encoder", "manifest_sha256", "selection_lock_sha256", "source_hashes"):
        if evaluation[key] != prescore[key]:
            raise ValueError("Benchmark index differs from pre-score receipt")
    audit.check_hash(manifest_path.parent / "ledger.json", prescore["ledger_file_sha256"])
    for name, expected in evaluation["source_hashes"].items():
        audit.check_hash(name, expected)
    expected = expected_states(manifest)
    runs = [{key: value for key, value in row.items() if key != "datasets"} for row in index["runs"]]
    if runs != expected or receipt["states"] != expected or index["state_count"] != len(runs) or index["archive_count"] != 2 * len(runs):
        raise ValueError("Diagnostic state inventory differs from locked selections")
    evaluated = {run["state_id"]: run for run in evaluation["runs"]}
    if len(evaluated) != len(evaluation["runs"]):
        raise ValueError("Duplicate benchmark state IDs")
    datasets, teachers, references = {}, {}, {}
    for name in DATASETS:
        if receipt["input_hashes"][name] != prescore["input_hashes"][name]:
            raise ValueError("Diagnostic gallery differs from benchmark gallery")
        datasets[name] = load_dataset(audit, config, name, receipt["input_hashes"][name])
        teachers[name] = forward_features(datasets[name][0])
        references[name] = checked_archive(audit, evaluated["frozen"]["datasets"][name])
    counts, strata = Counter(), defaultdict(Counter)
    unchanged_reference = {}
    for run in index["runs"]:
        if set(run["datasets"]) != set(DATASETS):
            raise ValueError("Missing certificate gallery")
        saved_state = {key: value for key, value in run.items() if key not in ("datasets", "selection_roles")}
        if saved_state != {key: value for key, value in evaluated[run["state_id"]].items() if key != "datasets"}:
            raise ValueError("Diagnostic checkpoint differs from benchmark state")
        audit.check_hash(run["checkpoint"], run["checkpoint_sha256"])
        checkpoint = torch.load(audit.path(run["checkpoint"]), map_location="cpu", weights_only=True)
        if checkpoint["ledger_sha256"] != manifest["ledger_sha256"] or checkpoint["protocol_sha256"] != PROTOCOL_SHA256:
            raise ValueError("Checkpoint provenance differs")
        for key in ("method", "epoch", "seed", "learning_rate"):
            if checkpoint[key] != run[key]:
                raise ValueError("Checkpoint metadata differs")
        weights = checkpoint["state_dict"]
        dimension = 512 if encoder == "vit_b32" else 1024
        if set(weights) != {"image.weight", "text.weight"} or any(value.shape != (dimension, dimension) or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Unexpected linear adapter weights")
        norm = float(np.sqrt(sum(np.square(value.numpy().astype(np.float64)).sum() for value in weights.values())))
        audit.equal(norm, run["update_norm"], "update_norm")
        stratum = "nonzero_update" if run["epoch"] > 0 and norm > 0 else "unchanged"
        if stratum == "unchanged" and norm != 0:
            raise ValueError("An epoch-zero checkpoint contains a learned update")
        counts[f"{stratum}_states"] += 1
        for name, record in run["datasets"].items():
            raw = checked_archive(audit, record)
            audit.check_hash(record["metadata"], record["metadata_sha256"])
            metadata = read_json(audit.path(record["metadata"]))
            if (metadata["predictions_sha256"] != record["predictions_sha256"] or metadata["summary"] != record["summary"] or
                    metadata["provenance"] != {"state": {key: value for key, value in run.items() if key != "datasets"},
                                               "dataset": name, "prediagnostic_receipt_sha256": index["prediagnostic_receipt_sha256"]}):
                raise ValueError("Diagnostic metadata provenance differs")
            features, relevance, identity = datasets[name]
            benchmark = checked_archive(audit, evaluated[run["state_id"]]["datasets"][name])
            for key, ids in identity.items():
                if any(not np.array_equal(ids, value[key]) for value in (raw, benchmark, references[name])):
                    raise ValueError("Diagnostic query or gallery identities differ")
            student = forward_features(features, weights)
            teacher = teachers[name]
            if stratum == "unchanged" and any(not np.array_equal(left, right) for left, right in zip(teacher, student, strict=True)):
                raise ValueError("An unchanged adapter altered normalized features")
            expected_keys = set(identity) | {f"{direction}_{key}" for direction in ("i2t", "t2i") for key in BOOLS + FLOATS + ("benchmark_teacher_ranks", "benchmark_student_ranks")}
            if set(raw) != expected_keys:
                raise ValueError("Diagnostic archive inventory differs")
            for direction, d in (("i2t", 0), ("t2i", 1)):
                saved = {key.removeprefix(direction + "_"): value for key, value in raw.items() if key.startswith(direction + "_")}
                if not np.array_equal(saved["benchmark_teacher_ranks"], references[name][f"{direction}_ranks"]) or not np.array_equal(saved["benchmark_student_ranks"], benchmark[f"{direction}_ranks"]):
                    raise ValueError("Diagnostic ranks differ from benchmark ranks")
                cache_key = name, direction
                if stratum == "unchanged" and cache_key in unchanged_reference:
                    # Every unchanged checkpoint has exact zero tensors and equal normalized features.
                    # Reuse only independently reconstructed teacher-versus-teacher query arrays.
                    compare_arrays(unchanged_reference[cache_key], saved, maxima)
                else:
                    pieces = defaultdict(list)
                    for start in range(0, len(teacher[d]), block_size):
                        stop = min(start + block_size, len(teacher[d]))
                        teacher_scores = teacher[d][start:stop] @ teacher[1 - d].T
                        student_scores = (teacher_scores if stratum == "unchanged" else
                                          student[d][start:stop] @ student[1 - d].T)
                        recomputed = independent_rows(teacher_scores, student_scores,
                            relevance[d][start:stop], receipt["native_logit_scale"])
                        compare_arrays(recomputed, {key: value[start:stop] for key, value in saved.items()}, maxima)
                        if stratum == "unchanged":
                            for key, values in recomputed.items():
                                pieces[key].append(values)
                    if stratum == "unchanged":
                        unchanged_reference[cache_key] = {key: np.concatenate(values) for key, values in pieces.items()}
                summary = independent_summary(saved)
                published = record["summary"][direction]
                if any(summary[key] != published[key] for key in summary):
                    raise ValueError("Independent coverage summary differs")
                for target in (strata["all"], strata[stratum]):
                    target.update({key: summary["counts"][key] for key in ("queries", "teacher_correct", "student_correct", "actually_retained")})
                    target.update({key: summary["certificate_coverage"][key]["certified_queries"] for key in MASKS})
                counts["state_query_observations"] += len(saved["teacher_correct"])
            counts["archives"] += 1
            print(json.dumps({"certificate_audit": encoder, "state": run["state_id"], "dataset": name, "status": "passed"}), flush=True)
    if index["false_exact_kl_certificates"] != 0:
        raise ValueError("Diagnostic index reports false certificates")
    return encoder, {"counts": dict(counts), "coverage_counts": {key: dict(value) for key, value in strata.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs=2, required=True)
    parser.add_argument("--dataset-configs", nargs=2, required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.protocol_sha256 != PROTOCOL_SHA256 or args.block_size < 1 or args.torch_threads < 1:
        raise ValueError("Invalid audit contract")
    audit, maxima = Audit(ROOT), defaultdict(float)
    audit.check_hash(args.protocol, PROTOCOL_SHA256)
    lock = read_json(audit.path(args.selection_lock))
    if lock["protocol_sha256"] != PROTOCOL_SHA256 or set(lock["encoders"]) != set(ENCODERS):
        raise ValueError("Both encoder selections must be locked")
    configs = {read_json(audit.path(path))["encoder"]: path for path in args.dataset_configs}
    if set(configs) != set(ENCODERS):
        raise ValueError("Both dataset configs are required")
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    results, lock_hash = {}, digest(audit.path(args.selection_lock))
    for path in args.indices:
        encoder = read_json(audit.path(path))["encoder"]
        if encoder in results or encoder not in ENCODERS:
            raise ValueError("Duplicate or unknown certificate index encoder")
        encoder, results[encoder] = audit_index(audit, path, configs[encoder], lock, lock_hash, args.block_size, maxima)
    if set(results) != set(ENCODERS):
        raise ValueError("Both certificate indices are required")
    inputs = args.indices + args.dataset_configs + [args.selection_lock, args.protocol]
    result = {"schema_version": 1, "status": "passed", "protocol_sha256": PROTOCOL_SHA256,
        "input_sha256": {str(audit.path(path).relative_to(ROOT)): digest(audit.path(path)) for path in inputs},
        "source_sha256": {name: digest(ROOT / name) for name in ("scripts/audit_retention_certificates_independent.py",
                           "scripts/audit_allocation_distillation_independent.py")},
        "independent_implementation": "Recomputes all full-gallery scores from cached features and checkpoint weights. Enumerates feasible positive-set pooling sizes. Imports no production adapter, rank, or certificate implementation.",
        "scope": "Same complete galleries only; forward teacher-to-student KL at T=2, without T-squared scaling.",
        "interpretation": "Counts are state-query observations over reused galleries. Nonzero-update coverage is separate from unchanged-state coverage. Coverage does not establish caption improvement or out-of-gallery guarantees.",
        "floating_point_caveat": "Independent floating-point recomputation, not an interval-arithmetic proof.",
        "false_certificates": 0, "encoders": results, "max_absolute_differences": dict(maxima),
        "checks": dict(audit.counts), "runtime": {"python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "torch_threads": args.torch_threads, "block_size": args.block_size}}
    output = audit.path(args.output)
    if output.exists() and read_json(output) != result:
        raise ValueError("Existing certificate audit receipt differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "passed", "encoders": list(results), "false_certificates": 0}), flush=True)


if __name__ == "__main__":
    main()
