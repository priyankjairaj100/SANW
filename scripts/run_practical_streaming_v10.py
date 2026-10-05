#!/usr/bin/env python3
"""Fit the single inherited v10 recipe on the locked train-only 6000-owner pool.

No validation selection, inner grid, official development, or benchmark scoring
occurs here. Replication requires the two prespecified seed-17 pilots to pass.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_practical_constrained_v8 import atomic_json
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_joint_v9 import JointCompositionExamples, JointFitConfig, fit_joint
from gcr.practical_no_retention_v10 import fit_joint_without_retention
from gcr.practical_streaming_v10 import FrozenScoreCache, StreamingFullGalleryConstraints, StreamingFullGallerySourceLoss
from gcr.practical_training_data_v10 import digest, load_training, verify_record


REQUIRED_SOURCES = (
    "scripts/run_practical_streaming_v10.py", "scripts/run_practical_constrained_v8.py",
    "src/gcr/practical_training_data_v10.py", "src/gcr/practical_streaming_v10.py",
    "src/gcr/practical_joint_v9.py", "src/gcr/practical_constrained_v8.py",
    "src/gcr/practical_constrained_evaluation_v8.py", "src/gcr/practical_no_retention_v10.py",
)


def verify_protocol(repository, protocol, encoder, seed):
    if (protocol.get("study") != "sanw_practical_v10"
            or sorted(protocol.get("encoders", [])) != ["rn50", "vit_b32"]
            or encoder not in protocol["encoders"] or protocol.get("seeds") != [17, 29, 43]
            or seed not in protocol["seeds"]):
        raise ValueError("Wrong study, encoder, or prescribed three-seed set")
    expected = asdict(JointFitConfig(radius=1.0, composition_weight=.25))
    expected.pop("seed")
    if protocol.get("fit_config") != expected:
        raise ValueError("V10 must retain the inherited single v9 recipe without a new grid")
    config = JointFitConfig(**protocol["fit_config"], seed=seed)
    config.validate()
    sources = protocol.get("source_sha256", {})
    if not set(REQUIRED_SOURCES) <= set(sources):
        raise ValueError("Protocol does not bind every required source")
    for name, sha in sources.items():
        if digest(Path(repository) / name) != sha:
            raise ValueError(f"Declared source changed: {name}")
    streaming = protocol.get("streaming", {})
    if set(streaming) != {"cache_block_size", "query_block_size", "fit_threads"}:
        raise ValueError("Streaming execution settings must be explicit")
    if any(type(streaming[key]) is not int or streaming[key] < 1 for key in streaming):
        raise ValueError("Streaming settings must be positive integers")
    thread_values = {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if any(value != str(streaming["fit_threads"]) for value in thread_values.values()):
        raise ValueError("Set all three BLAS/OMP thread environment values to the protocol fit_threads before launching")
    return config, thread_values


def verify_replication_gate(repository, gate_path, protocol_sha):
    gate = json.loads(Path(gate_path).read_text())
    if (gate.get("study") != "sanw_practical_v10_replication_gate" or gate.get("family") != "joint"
            or gate.get("passed") is not True or gate.get("protocol_sha256") != protocol_sha
            or sorted(gate.get("encoders_passed", [])) != ["rn50", "vit_b32"] or gate.get("seed") != 17):
        raise ValueError("Both seed-17 official-development pilots must pass before replication")
    runs = gate.get("pilot_runs", [])
    if sorted(item.get("encoder", "") for item in runs) != ["rn50", "vit_b32"]:
        raise ValueError("Gate must bind both pilot results and checkpoints")
    for item in runs:
        paths = {key: verify_record(repository, item[key]) for key in ("completion", "checkpoint", "result")}
        completion, result = (json.loads(paths[key].read_text()) for key in ("completion", "result"))
        if (completion.get("study") != "sanw_practical_v10" or completion.get("encoder") != item["encoder"]
                or completion.get("protocol_sha256") != protocol_sha or completion.get("config", {}).get("seed") != 17
                or completion.get("selected_checkpoint", {}).get("sha256") != item["checkpoint"]["sha256"]
                or result.get("study") != "sanw_practical_v10_official_development"
                or result.get("passed") is not True or result.get("encoder") != item["encoder"]
                or result.get("seed") != 17 or result.get("protocol_sha256") != protocol_sha
                or result.get("checkpoint_sha256") != item["checkpoint"]["sha256"]):
            raise ValueError("Replication gate pilot identities differ")
    return {"path": str(Path(gate_path).resolve()), "sha256": digest(gate_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--family", choices=("joint", "no_retention"), default="joint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--replication-gate", type=Path)
    parser.add_argument("--control-gate", type=Path)
    args = parser.parse_args()
    repository, output = args.repository.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing fit")
    protocol = json.loads(args.protocol.read_text())
    config, thread_values = verify_protocol(repository, protocol, args.encoder, args.seed)
    protocol_sha = digest(args.protocol)
    replication = None
    control_gate = None
    if args.family == "no_retention":
        expected_control = {"execution": "only_after_joint_seed17_development_pass", "config": "same_inherited_joint_config",
                            "seeds": [17, 29, 43]}
        if protocol.get("controls", {}).get("no_retention") != expected_control or args.control_gate is None:
            raise ValueError("The matched control must be prespecified and conditional on the passing joint pilot gate")
        if args.replication_gate is not None:
            raise ValueError("No-retention control uses its explicit control gate")
        control_gate = verify_replication_gate(repository, args.control_gate, protocol_sha)
    elif args.control_gate is not None:
        raise ValueError("Joint candidate does not use a control gate")
    elif args.seed != 17:
        if args.replication_gate is None:
            raise ValueError("Replication requires the passing two-encoder seed-17 gate")
        replication = verify_replication_gate(repository, args.replication_gate, protocol_sha)
    elif args.replication_gate is not None:
        raise ValueError("Seed-17 pilots do not use a replication gate")
    import torch
    torch.set_num_threads(1)
    start = time.monotonic()
    images, texts, source_rows, owner, sources, supported, contra, provenance, scale = load_training(repository, args.encoder, protocol)
    scorer = ConstrainedBilinearScorer.from_training(images, texts, config.rank)
    settings = protocol["streaming"]
    cache = FrozenScoreCache.create(args.cache.resolve(), images, texts[source_rows], owner, settings["cache_block_size"])
    constraints = StreamingFullGalleryConstraints(images, texts[source_rows], owner, scorer, cache,
                                                  config.retention_fraction, settings["cache_block_size"])
    composition = JointCompositionExamples(images, texts, sources, supported, contra, scorer)
    retrieval = StreamingFullGallerySourceLoss(constraints, scale, settings["query_block_size"])
    identity = {"study": "sanw_practical_v10", "family": args.family, "mode": "full", "encoder": args.encoder,
                "config": asdict(config), "protocol_sha256": protocol_sha,
                "protocol": {"path": str(args.protocol.resolve()), "sha256": protocol_sha},
                "replication_gate": replication, "control_gate": control_gate, "source_sha256": protocol["source_sha256"],
                "training_provenance": provenance, "retrieval_logit_scale": scale,
                "fit_gallery_image_count": len(images), "fit_gallery_text_count": len(source_rows),
                "frozen_score_cache": {"path": str(args.cache.resolve()), "metadata_sha256": digest(args.cache / "metadata.json")},
                "streaming": settings, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                                         "torch": torch.__version__, "blas_threads": thread_values,
                                                         "torch_normalization_threads": 1},
                "state_identity": "checkpoint_file_sha256; floating_derived_norms_are_descriptive_only",
                "normalization": "one_Torch_float32_L2_pass_then_float64_canonical_score",
                "official_development_or_benchmarks_used": False}
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    atomic_json(output / "ledger.json", {"identity": identity, "ledger_sha256": ledger_sha})
    checkpoint_rows = []

    def on_epoch(row, coefficient):
        scorer.coefficient = coefficient
        path = output / "checkpoints" / f"epoch_{row['epoch']:03d}.npz"
        scorer.save(path)
        item = dict(row)
        item["checkpoint"] = {"path": str(path.relative_to(output)), "sha256": digest(path), "ledger_sha256": ledger_sha}
        checkpoint_rows.append(item)
        atomic_json(output / "history.json", checkpoint_rows)
        print(json.dumps({"epoch": row["epoch"], "objective": row["training_objective"], "retrieval_loss": row["retrieval_loss"],
                          "joint_accuracy": row["composition"]["image_mean_joint_accuracy"],
                          "coefficient_norm_descriptive": float(np.sqrt(np.sum(coefficient * coefficient, dtype=np.float64))),
                          "objective_gap_upper": row["ball_relaxed_convex_suboptimality_upper_bound"]}), flush=True)

    fitter = fit_joint if args.family == "joint" else fit_joint_without_retention
    result = fitter(scorer, composition, retrieval, constraints, config, on_epoch)
    if not np.any(scorer.coefficient != 0):
        raise ValueError("Selected state must be nonzero")
    if args.family == "joint":
        certificate = result["final_certificate"]
        if (certificate.get("ranking_checked_canonically") is not True or certificate.get("ranking_preserved") is not True
                or certificate.get("feasible_with_tolerance") is not True):
            raise ValueError("Selected candidate must preserve canonical finite-training rankings")
    elif (result.get("retention_enforced") is not False
          or result.get("final_training_retention_diagnostic", {}).get("ranking_checked_canonically") is not True):
        raise ValueError("Control must measure canonical retention without imposing it")
    scorer.save(output / "selected.npz")
    result.update({"study": "sanw_practical_v10", "family": args.family, "mode": "full", "encoder": args.encoder,
                   "ledger_sha256": ledger_sha, "protocol_sha256": protocol_sha,
                   "selected_checkpoint": {"path": "selected.npz", "sha256": digest(output / "selected.npz")},
                   "checkpoint_history": checkpoint_rows, "elapsed_seconds": time.monotonic() - start})
    atomic_json(output / "completion.json", result)
    print(json.dumps({"complete": True, "selected_epoch": result["selected_epoch"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
