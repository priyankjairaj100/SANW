#!/usr/bin/env python3
"""Recompute training-only objective and retention for one completed v10 state.

This diagnostic does not optimize, alter a state, select a new checkpoint, or
load any development/confirmation/benchmark outcomes. Run sequentially after
the large fitting jobs to avoid competing for memory and CPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_practical_streaming_v10 import verify_protocol
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_joint_v9 import JointCompositionExamples
from gcr.practical_streaming_v10 import (
    FrozenScoreCache, StreamingFullGalleryConstraints, StreamingFullGallerySourceLoss, cached_exact_retrieval,
)
from gcr.practical_training_data_v10 import digest, load_training


def record(path):
    return {"path": str(path.resolve()), "sha256": digest(path)}


def close(actual, expected, name, atol=2e-11):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=2e-10, abs_tol=atol):
        raise ValueError(f"Recomputed {name} differs from the saved training result")
    return float(actual-expected)


def paired_summary(before, after):
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    return {"queries": len(before), "frozen_correct": int(before.sum()), "selected_correct": int(after.sum()),
            "wins": int((~before & after).sum()), "losses": int((before & ~after).sum()),
            "frozen_r1": float(before.mean()), "selected_r1": float(after.mean()),
            "paired_gain_pp": float(np.mean(after.astype(float)-before)*100)}


def diagnose(repository, protocol_path, completion_path, cache_override=None):
    started = time.monotonic()
    repository, protocol_path, completion_path = map(Path, (repository, protocol_path, completion_path))
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(completion_path.read_text())
    encoder, family, seed = result.get("encoder"), result.get("family"), result.get("config", {}).get("seed")
    config, thread_values = verify_protocol(repository, protocol, encoder, seed)
    if (result.get("study") != "sanw_practical_v10" or result.get("mode") != "full"
            or family not in ("joint", "no_retention") or result.get("protocol_sha256") != digest(protocol_path)
            or result.get("config") != asdict(config)):
        raise ValueError("Completion identity differs from the locked v10 fit")
    directory = completion_path.parent
    ledger_path = directory / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    identity = ledger["identity"]
    expected_ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (ledger.get("ledger_sha256") != expected_ledger_sha or result.get("ledger_sha256") != expected_ledger_sha
            or identity.get("protocol_sha256") != digest(protocol_path) or identity.get("config") != asdict(config)
            or identity.get("encoder") != encoder or identity.get("family") != family
            or identity.get("source_sha256") != protocol["source_sha256"]
            or identity.get("fit_gallery_image_count") != 6000 or identity.get("fit_gallery_text_count") != 30000):
        raise ValueError("Ledger identity differs")
    history, selected_epoch = result["history"], result["selected_epoch"]
    expected_steps = config.epochs*((6000+config.batch_size-1)//config.batch_size)
    if ([row["epoch"] for row in history] != list(range(1, config.epochs+1))
            or result["optimizer_steps"] != expected_steps
            or any(row["optimizer_steps"] != row["epoch"]*((6000+config.batch_size-1)//config.batch_size) for row in history)
            or any(not math.isfinite(row["training_objective"]) for row in history)):
        raise ValueError("Locked optimizer budget was not completed")
    checkpoint_history = result["checkpoint_history"]
    if [{key: value for key, value in row.items() if key != "checkpoint"} for row in checkpoint_history] != history:
        raise ValueError("Checkpoint histories disagree")
    eligible = [row for row in history if row["nonzero"]]
    best = min(eligible, key=lambda row: (row["training_objective"], row["epoch"]))
    if selected_epoch != best["epoch"]:
        raise ValueError("Saved selection differs from minimum nonzero training objective")
    if family == "joint" and any(not row["certificate"]["ranking_preserved"] or not row["certificate"]["feasible_with_tolerance"] for row in history):
        raise ValueError("A supposedly feasible joint checkpoint failed its saved training certificate")
    selected_path = directory / result["selected_checkpoint"]["path"]
    if digest(selected_path) != result["selected_checkpoint"]["sha256"]:
        raise ValueError("Selected state hash differs")
    epoch_record = next(row["checkpoint"] for row in result["checkpoint_history"] if row["epoch"] == selected_epoch)
    epoch_path = directory / epoch_record["path"]
    if digest(epoch_path) != epoch_record["sha256"] or epoch_record["ledger_sha256"] != expected_ledger_sha:
        raise ValueError("Selected epoch state provenance differs")
    with np.load(selected_path, allow_pickle=False) as a, np.load(epoch_path, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Selected state differs from its recorded training epoch")
    model = ConstrainedBilinearScorer.load(selected_path)
    coefficient = model.coefficient.copy()
    if not np.any(coefficient != 0) or coefficient.shape != (config.rank, config.rank) or not np.isfinite(coefficient).all():
        raise ValueError("Wrong or zero selected coefficient")
    import torch
    torch.set_num_threads(1)
    images, texts, source_rows, owner, sources, supported, contra, provenance, scale = load_training(repository, encoder, protocol)
    if provenance != identity["training_provenance"] or scale != identity["retrieval_logit_scale"]:
        raise ValueError("Reconstructed training data/provenance differ")
    if not np.allclose(model.image_mean, images.mean(axis=0), rtol=0, atol=1e-14) or not np.allclose(model.text_mean, texts.mean(axis=0), rtol=0, atol=1e-14):
        raise ValueError("Serialized PCA means differ from training-only means")
    cache_path = Path(cache_override) if cache_override else Path(identity["frozen_score_cache"]["path"])
    if digest(cache_path/"metadata.json") != identity["frozen_score_cache"]["metadata_sha256"]:
        raise ValueError("Frozen cache metadata differs from the fit")
    cache = FrozenScoreCache.open(cache_path, images, texts[source_rows], owner)
    settings = protocol["streaming"]
    constraints = StreamingFullGalleryConstraints(images, texts[source_rows], owner, model, cache,
                                                  config.retention_fraction, settings["cache_block_size"])
    composition = JointCompositionExamples(images, texts, sources, supported, contra, model)
    retrieval = StreamingFullGallerySourceLoss(constraints, scale, settings["query_block_size"])
    zero = np.zeros_like(coefficient)
    initial_r, initial_rg = retrieval.loss_gradient(zero)
    initial_c, initial_cg = composition.loss_gradient(zero, composition.eligible, config)
    multiplier = config.composition_weight*np.linalg.norm(initial_rg)/max(np.linalg.norm(initial_cg), 1e-12)
    differences = {"initial_retrieval": close(initial_r, result["initial_retrieval_loss"], "initial retrieval loss"),
                   "initial_joint": close(initial_c, result["initial_joint_composition_loss"], "initial joint loss"),
                   "fixed_multiplier": close(multiplier, result["fixed_composition_multiplier"], "fixed multiplier")}
    selected_r, selected_rg = retrieval.loss_gradient(coefficient)
    selected_c, selected_cg = composition.loss_gradient(coefficient, composition.eligible, config)
    norm = float(np.sqrt(np.sum(coefficient*coefficient, dtype=np.float64)))
    if norm > config.radius*(1+64*np.finfo(np.float64).eps):
        raise ValueError("Selected coefficient violates the radius needed for the objective bound")
    objective = selected_r + multiplier*selected_c + config.ridge*norm**2/2
    initial_objective = initial_r + multiplier*initial_c
    gradient = selected_rg + multiplier*selected_cg + config.ridge*coefficient
    gap_bound = max(0., float(np.sum(gradient*coefficient)+config.radius*np.linalg.norm(gradient)))
    differences.update({"selected_objective": close(objective, result["selected_training_objective"], "selected objective"),
                        "initial_objective": close(initial_objective, result["initial_training_objective"], "initial objective"),
                        "upper_gap": close(gap_bound, best["ball_relaxed_convex_suboptimality_upper_bound"], "convex gap bound")})
    certificate = constraints.scan(coefficient, config.feasibility_tolerance, add=False, canonical=True)
    saved_certificate = result["final_certificate"] if family == "joint" else result["final_training_retention_diagnostic"]
    for key, value in certificate.items():
        if key == "active_constraints":
            continue  # Reconstructed diagnostics intentionally have no fitted active-set state.
        if key == "min_constraint_slack" and value is not None:
            close(value, saved_certificate[key], "minimum constraint slack", atol=2e-12)
        elif value != saved_certificate[key]:
            raise ValueError(f"Final training-retention diagnostic differs on {key}")
    if family == "joint" and not (certificate["ranking_preserved"] and certificate["feasible_with_tolerance"]):
        raise ValueError("Selected joint state does not retain the promised finite-training scope")
    selected_raw = cached_exact_retrieval(images, texts[source_rows], owner, cache, model, settings["query_block_size"])
    retention = {}
    for group, image_mask in (("all6000", np.ones(len(images), dtype=bool)),
                              ("original1200", np.arange(len(images)) < 1200),
                              ("added4800", np.arange(len(images)) >= 1200)):
        retention[group] = {"i2t": paired_summary(constraints.protect_i2t[image_mask], selected_raw["i2t_correct"][image_mask]),
                            "t2i": paired_summary(constraints.protect_t2i[image_mask[owner]], selected_raw["t2i_correct"][image_mask[owner]])}
    reduction = float(initial_objective-objective)
    last = history[-8:]
    cosine_denominator = np.linalg.norm(selected_rg)*np.linalg.norm(selected_cg)
    return {"study": "sanw_practical_v10_postfit_training_diagnosis", "family": family, "encoder": encoder, "seed": seed,
            "scope": "completed_selected_state_training_only_no_updates_no_checkpoint_reselection_no_heldout_outcomes",
            "protocol": record(protocol_path), "completion": record(completion_path), "ledger": record(ledger_path),
            "selected_checkpoint": record(selected_path), "selected_epoch_checkpoint": record(epoch_path),
            "diagnostic_source": record(Path(__file__)), "source_hashes_verified": True,
            "environment": {"numpy": np.__version__, "blas_threads": thread_values, "torch_normalization_threads": 1},
            "selected_epoch": selected_epoch, "epochs": len(history), "optimizer_steps": expected_steps,
            "original_fit_elapsed_seconds": result["elapsed_seconds"], "gallery_images": len(images), "gallery_texts": len(source_rows),
            "eligible_joint_images": len(composition.eligible), "coefficient_frobenius_norm": norm,
            "radius_utilization": norm/config.radius, "fixed_composition_multiplier": float(multiplier),
            "initial_objective": float(initial_objective), "selected_objective": float(objective), "achieved_objective_reduction": reduction,
            "selected_retrieval_loss": float(selected_r), "selected_joint_loss": float(selected_c),
            "recomputed_full_gradient_norm": float(np.linalg.norm(gradient)), "ball_relaxed_suboptimality_upper_bound": gap_bound,
            "gap_divided_by_achieved_reduction": gap_bound/reduction if reduction > 0 else None,
            "selected_retrieval_joint_gradient_cosine": float(np.sum(selected_rg*selected_cg)/cosine_denominator) if cosine_denominator > 0 else None,
            "last_eight_objective_range": max(row["training_objective"] for row in last)-min(row["training_objective"] for row in last),
            "last_epoch_minus_selected_objective": history[-1]["training_objective"]-result["selected_training_objective"],
            "recomputed_minus_recorded": differences, "selected_composition": composition.summary(coefficient),
            "frozen_composition": composition.summary(zero), "canonical_training_retention": retention,
            "finite_training_constraint_diagnostic": certificate,
            "stored_selected_active_constraint_count": saved_certificate["active_constraints"],
            "minimum_saved_radial_restoration_factor": min(row["certificate"]["radial_restoration_factor"] for row in history) if family == "joint" else None,
            "certificate_scope": "Only the specified finite training galleries and serialized canonical pair scorer; no unseen-data retention guarantee.",
            "gap_interpretation": "Convex first-order upper bound via relaxation to the Frobenius ball. A small bound certifies training-objective proximity; a large upper bound does not prove poor optimization.",
            "training_subgroups_scope": "All subgroups retain the same complete6000-image/30000-source-caption gallery; only reported queries are partitioned.",
            "practical_or_paper_success_inferred": False, "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "diagnostic_elapsed_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an earlier postfit diagnosis")
    result = diagnose(args.repository.resolve(), args.protocol.resolve(), args.completion.resolve(), args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("encoder", "family", "seed", "selected_epoch", "optimizer_steps",
                                                  "gap_divided_by_achieved_reduction", "diagnostic_elapsed_seconds")}))


if __name__ == "__main__":
    main()
