#!/usr/bin/env python3
"""Independently audit completed v8 fits using training rows only.

This checks every saved checkpoint, the declared selection, training geometry,
and all source-gallery scores through explicit canonical pair enumeration.
It neither fits a model nor loads a development retrieval gallery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from run_practical_constrained_v8 import load_training
from evaluate_practical_constrained_development_v8 import verified_run


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def independent_coordinates(features, mean, basis):
    return np.einsum("nd,dr->nr", features - mean, basis, optimize=False)


def independent_full_gallery(images, texts, model):
    """Exhaustive rowwise pair products; no retrieval screener or rank helper."""
    x = independent_coordinates(images, model.image_mean, model.image_basis)
    y = independent_coordinates(texts, model.text_mean, model.text_basis)
    xa = np.einsum("nd,dr->nr", x, model.coefficient, optimize=False)
    frozen = np.empty((len(images), len(texts)), dtype=np.float64)
    residual = np.empty_like(frozen)
    for i in range(len(images)):
        frozen[i] = np.sum(images[i] * texts, axis=1, dtype=np.float64)
        residual[i] = np.sum(xa[i] * y, axis=1, dtype=np.float64)
    return frozen, residual


def audit_one(run, protocol_hash):
    start = time.monotonic()
    identity, completion, protocol, model, norm = verified_run(run, protocol_hash)
    images, texts, source_rows, owner, positive, negative, provenance = load_training(ROOT, identity["encoder"], identity["config"]["relation_type"])
    assert provenance == identity["training_provenance"]
    assert len(images) == 1200 and len(source_rows) == 6000
    mean_error = max(float(np.max(np.abs(images.mean(axis=0) - model.image_mean))),
                     float(np.max(np.abs(texts.mean(axis=0) - model.text_mean))))
    assert mean_error < 1e-14
    pca_checks = []
    for features, mean, basis in ((images, model.image_mean, model.image_basis), (texts, model.text_mean, model.text_basis)):
        centered = features - mean
        covariance = centered.T @ centered
        eigenvalues = np.einsum("dr,ds,sr->r", basis, covariance, basis, optimize=True)
        pca_error = float(np.max(np.abs(covariance @ basis - basis * eigenvalues)))
        orthogonality_error = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
        assert pca_error < 1e-8 and orthogonality_error < 1e-10
        assert np.all(np.diff(eigenvalues) <= 1e-9)
        all_eigenvalues = np.linalg.eigvalsh(covariance)[::-1][:basis.shape[1]]
        eigenvalue_error = float(np.max(np.abs(eigenvalues - all_eigenvalues)))
        assert eigenvalue_error < 1e-8
        pca_checks.append({"rows": len(features), "orthogonality_max_error": orthogonality_error,
                           "covariance_eigenvector_max_error": pca_error, "leading_eigenvalue_max_error": eigenvalue_error})
    checkpoint_records = []
    for row in completion["checkpoint_history"]:
        cp = run / row["checkpoint"]["path"]
        assert digest(cp) == row["checkpoint"]["sha256"]
        state = ConstrainedBilinearScorer.load(cp)
        for attribute in ("image_mean", "text_mean", "image_basis", "text_basis"):
            assert np.array_equal(getattr(state, attribute), getattr(model, attribute))
        state_norm = float(np.linalg.norm(state.coefficient))
        assert row["nonzero"] == bool(np.any(state.coefficient != 0))
        assert state_norm <= protocol["fit_config"]["radius"] * (1 + 64 * np.finfo(np.float64).eps)
        certificate = row["certificate"]
        assert certificate["ranking_checked_canonically"] and certificate["ranking_preserved"] and certificate["feasible_with_tolerance"]
        assert certificate["lost_frozen_correct_i2t"] == certificate["lost_frozen_correct_t2i"] == certificate["violated_constraints"] == 0
        assert state_norm == certificate["coefficient_frobenius_norm"]
        checkpoint_records.append({"epoch": row["epoch"], "sha256": digest(cp), "norm": state_norm})
    assert json.loads((run / "history.json").read_text()) == completion["checkpoint_history"]
    frozen, residual = independent_full_gallery(images, texts[source_rows], model)
    trained = frozen + residual
    owned = np.arange(len(images))[:, None] == owner[None, :]
    frozen_i = frozen.argmax(axis=1)
    frozen_t = frozen.argmax(axis=0)
    protected_i = owner[frozen_i] == np.arange(len(images))
    protected_t = frozen_t == owner
    trained_i = trained.argmax(axis=1)
    trained_t = trained.argmax(axis=0)
    best_positive = np.where(owned, frozen, -np.inf).argmax(axis=1)
    gamma = protocol["fit_config"]["retention_fraction"]
    i_slack = ((1-gamma) * (frozen[np.arange(len(images)), best_positive, None] - frozen)
               + residual[np.arange(len(images)), best_positive, None] - residual)
    t_slack = ((1-gamma) * (frozen[owner, np.arange(len(owner))][None, :] - frozen)
               + residual[owner, np.arange(len(owner))][None, :] - residual)
    i_valid = ~owned & protected_i[:, None]
    t_valid = ~owned & protected_t[None, :]
    constraints_count = int(i_valid.sum() + t_valid.sum())
    minimum = min(float(i_slack[i_valid].min()), float(t_slack[t_valid].min()))
    violations = int((i_slack[i_valid] < -1e-12).sum() + (t_slack[t_valid] < -1e-12).sum())
    lost_i = int((protected_i & (owner[trained_i] != np.arange(len(images)))).sum())
    lost_t = int((protected_t & (trained_t != owner)).sum())
    counts = {"protected_i2t_queries": int(protected_i.sum()), "protected_t2i_queries": int(protected_t.sum()),
              "checked_constraints": constraints_count, "violated_constraints": violations,
              "lost_frozen_correct_i2t": lost_i, "lost_frozen_correct_t2i": lost_t,
              "current_i2t_correct": int((owner[trained_i] == np.arange(len(images))).sum()),
              "current_t2i_correct": int((trained_t == owner).sum())}
    assert violations == lost_i == lost_t == 0
    cert = completion["final_certificate"]
    assert all(cert[k] == v for k, v in counts.items()), (cert, counts)
    slack_difference = abs(minimum - cert["min_constraint_slack"])
    assert slack_difference < 1e-12
    # Independently recompute the selected scalar training objective.
    x = independent_coordinates(images, model.image_mean, model.image_basis)
    y = independent_coordinates(texts, model.text_mean, model.text_basis)
    xa = np.einsum("nd,dr->nr", x, model.coefficient, optimize=False)
    losses = []
    accuracies = []
    for i, (p, n) in enumerate(zip(positive, negative)):
        if not len(p) or not len(n):
            continue
        ps = np.sum(images[i] * texts[p], axis=1) + np.sum(xa[i] * y[p], axis=1)
        ns = np.sum(images[i] * texts[n], axis=1) + np.sum(xa[i] * y[n], axis=1)
        gaps = ps[:, None] - ns[None, :]
        z = (protocol["fit_config"]["composition_margin"] - gaps) / protocol["fit_config"]["temperature"]
        losses.append(float(np.logaddexp(0, z).mean()))
        accuracies.append(float((gaps > 0).mean()))
    objective = float(np.mean(losses) + protocol["fit_config"]["ridge"] * np.sum(model.coefficient ** 2) / 2)
    objective_error = abs(objective - completion["selected_training_objective"])
    assert objective_error < 1e-11
    selected_row = completion["history"][completion["selected_epoch"]-1]
    accuracy_error = abs(float(np.mean(accuracies)) - selected_row["composition"]["image_mean_pair_accuracy"])
    assert accuracy_error < 1e-12
    expected_steps = protocol["fit_config"]["epochs"] * int(np.ceil(len(losses) / protocol["fit_config"]["batch_size"]))
    assert completion["optimizer_steps"] == expected_steps
    return {"encoder": identity["encoder"], "seed": identity["config"]["seed"], "passed": True,
            "run": str(run.relative_to(ROOT)), "completion_sha256": digest(run / "completion.json"),
            "ledger_sha256": digest(run / "ledger.json"), "selected_checkpoint_sha256": digest(run / "selected.npz"),
            "selected_epoch": completion["selected_epoch"], "coefficient_norm": norm,
            "optimizer_steps": expected_steps, "checkpoint_count": len(checkpoint_records),
            "checkpoint_records": checkpoint_records, "training_mean_max_error": mean_error, "pca_checks": pca_checks,
            "independent_selected_certificate": dict(counts, min_constraint_slack=minimum),
            "max_minimum_slack_difference": slack_difference, "independent_training_objective": objective,
            "training_objective_absolute_error": objective_error, "training_pair_accuracy_absolute_error": accuracy_error,
            "full_gallery_canonical_pairs_evaluated": int(frozen.size),
            "heldout_benchmark_access": False, "development_gallery_access": False,
            "elapsed_seconds": time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = args.protocol.resolve()
    runs = [x.resolve() for x in args.runs]
    records = []
    for run in runs:
        record = audit_one(run, digest(protocol))
        records.append(record)
        print(json.dumps({k: record[k] for k in ("encoder", "passed", "selected_epoch", "coefficient_norm", "independent_selected_certificate", "elapsed_seconds")}), flush=True)
    result = {"schema": "sanw_v8_independent_completed_fit_audit_v1", "passed": True,
              "meaning_of_passed": "training_artifact_integrity_and_selected_finite_training_certificate_only_not_practical_success",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": {"path": str(protocol.relative_to(ROOT)), "sha256": digest(protocol)},
              "audit_source_sha256": digest(Path(__file__)), "fits": records,
              "total_checkpoint_count": sum(r["checkpoint_count"] for r in records),
              "total_full_gallery_pairs": sum(r["full_gallery_canonical_pairs_evaluated"] for r in records),
              "total_constraint_count": sum(r["independent_selected_certificate"]["checked_constraints"] for r in records),
              "scope": "Every epoch artifact/hash/header checked. Selected full source-gallery rankings and every declared inequality independently recomputed. Epoch rankings not individually rerun.",
              "no_fit_performed": True, "no_development_or_heldout_evaluation_performed": True}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"audit": str(args.output), "sha256": digest(args.output), "passed": True}), flush=True)


if __name__ == "__main__":
    main()
