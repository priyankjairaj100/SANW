#!/usr/bin/env python3
"""Evaluate one training-objective-selected v8 state on development data only.

The immutable protocol must bind this source, every input, and the exact gate.
This command cannot select another checkpoint, tune alpha, or load a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import (
    CanonicalScorer, composition_metrics, development_gate, exact_retrieval,
    paired_cluster_bootstrap,
)
from gcr.practical_training import input_paths
from gcr.review_training import SourceRetrievalPool
from gcr.training import FeatureDataset


CONTRACT = {
    "bootstrap_replicates": 100000, "bootstrap_seed": 20261007, "family_size": 80,
    "original_joint_gain": "strictly_positive", "source_pair_joint_gain": "nonnegative",
    "retrieval_mean_changes": "both_nonnegative",
    "retrieval_adjusted_lower_bounds": "both_strictly_greater_than_-0.01",
    "selection": "one_training_objective_selected_state_no_development_search",
}
SOURCES = (
    "scripts/evaluate_practical_constrained_development_v8.py",
    "src/gcr/practical_constrained_evaluation_v8.py", "src/gcr/practical_constrained_v8.py",
    "src/gcr/practical_training.py", "src/gcr/training.py", "src/gcr/review_training.py",
)


def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def root_path(value):
    filename = Path(value)
    filename = (filename if filename.is_absolute() else ROOT / filename).resolve()
    if not filename.is_relative_to(ROOT):
        raise ValueError("Artifact path escapes the repository")
    return filename


def read(value):
    return json.loads(root_path(value).read_text())


def record(value):
    filename = root_path(value)
    return {"path": str(filename.relative_to(ROOT)), "sha256": digest(filename), "bytes": filename.stat().st_size}


def verify_record(value):
    filename = root_path(value["path"])
    if digest(filename) != value["sha256"]:
        raise ValueError(f"Bound artifact changed: {filename}")
    return filename


def write_json(filename, value):
    filename = root_path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return record(filename)


def write_npz(filename, arrays):
    filename = root_path(filename)
    with filename.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return record(filename)


def verified_run(run, protocol_hash):
    run = root_path(run)
    ledger, completion = read(run / "ledger.json"), read(run / "completion.json")
    identity = ledger["identity"]
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(identity_bytes).hexdigest() != ledger["ledger_sha256"]:
        raise ValueError("Invalid training ledger identity hash")
    if identity["study"] != "sanw_constrained_bilinear_v8" or identity["mode"] != "fit" or identity["train_only"] is not True:
        raise ValueError("Require a completed v8 train-only fit")
    if completion["ledger_sha256"] != ledger["ledger_sha256"] or completion["development_or_test_used"] is not False:
        raise ValueError("Completion provenance is invalid")
    protocol_path = verify_record(identity["protocol"])
    if digest(protocol_path) != protocol_hash:
        raise ValueError("Protocol differs from the caller's pinned protocol hash")
    protocol = read(protocol_path)
    if protocol.get("study") != "sanw_constrained_bilinear_v8" or protocol.get("development_contract") != CONTRACT:
        raise ValueError("Development contract differs from this evaluator")
    if identity["encoder"] not in protocol.get("encoders", []):
        raise ValueError("Encoder is not declared in the protocol")
    source_hashes = protocol.get("source_sha256", {})
    if not set(SOURCES) <= set(source_hashes):
        raise ValueError("Protocol does not bind all evaluator and loader sources")
    for source, expected in source_hashes.items():
        if digest(root_path(source)) != expected:
            raise ValueError(f"Protocol-bound source changed: {source}")
    for source, expected in identity["source_sha256"].items():
        if digest(root_path(source)) != expected:
            raise ValueError(f"Training source changed: {source}")
    config = dict(identity["config"])
    seed = config.pop("seed")
    if protocol["fit_config"] != config or seed not in protocol["seeds"]:
        raise ValueError("Training settings differ from the frozen protocol")
    if completion["config"] != identity["config"] or completion["encoder"] != identity["encoder"]:
        raise ValueError("Completion and ledger disagree")
    if completion["selection"] != "minimum_feasible_nonzero_training_objective_then_earliest_epoch":
        raise ValueError("Unrecognized checkpoint selection rule")
    rows = completion["history"]
    if [r["epoch"] for r in rows] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Fit did not finish its complete fixed training budget")
    if any(not np.isfinite(row["training_objective"]) for row in rows):
        raise ValueError("Nonfinite training selection objective")
    eligible = [row for row in rows if row["nonzero"]]
    if not eligible:
        raise ValueError("No nonzero trained candidate")
    selected = min(eligible, key=lambda row: (row["training_objective"], row["epoch"]))
    if selected["epoch"] != completion["selected_epoch"] or selected["training_objective"] != completion["selected_training_objective"]:
        raise ValueError("Checkpoint was not selected solely by the declared training objective")
    certificate = completion["final_certificate"]
    if (not certificate["ranking_preserved"] or not certificate["feasible_with_tolerance"]
            or certificate.get("ranking_checked_canonically") is not True):
        raise ValueError("Selected state lacks its full training-gallery feasibility certificate")
    checkpoint = root_path(run / completion["selected_checkpoint"]["path"])
    if not checkpoint.is_relative_to(run) or digest(checkpoint) != completion["selected_checkpoint"]["sha256"]:
        raise ValueError("Selected checkpoint identity changed")
    model = ConstrainedBilinearScorer.load(checkpoint)
    norm = float(np.linalg.norm(model.coefficient))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Selected correction is not a finite nonzero trained update")
    if norm > config["radius"] * (1 + 64 * np.finfo(np.float64).eps):
        raise ValueError("Selected coefficient exceeds the declared Frobenius radius")
    # Compare arrays with the recorded selected epoch, not only a claimed header.
    checkpoint_rows = completion["checkpoint_history"]
    if [r["epoch"] for r in checkpoint_rows] != [r["epoch"] for r in rows]:
        raise ValueError("Checkpoint trajectory does not cover the fixed budget")
    for plain, checkpoint_row in zip(rows, checkpoint_rows):
        if {key: value for key, value in checkpoint_row.items() if key != "checkpoint"} != plain:
            raise ValueError("Checkpoint trajectory and selection history disagree")
    selected_row = checkpoint_rows[selected["epoch"] - 1]
    epoch_checkpoint = root_path(run / selected_row["checkpoint"]["path"])
    if not epoch_checkpoint.is_relative_to(run) or digest(epoch_checkpoint) != selected_row["checkpoint"]["sha256"]:
        raise ValueError("Training-selected epoch checkpoint changed")
    epoch_model = ConstrainedBilinearScorer.load(epoch_checkpoint)
    for attribute in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient"):
        if not np.array_equal(getattr(model, attribute), getattr(epoch_model, attribute)):
            raise ValueError("Selected checkpoint arrays differ from the training-selected epoch")
    return identity, completion, protocol, model, norm


def load_development(identity, protocol):
    encoder = identity["encoder"]
    dimension = {"vit_b32": 512, "rn50": 1024}[encoder]
    expected_paths = input_paths(ROOT, encoder)
    bound_inputs = protocol["development_inputs"][encoder]
    if set(bound_inputs) != set(expected_paths):
        raise ValueError("Protocol must bind exactly the six train/development input files")
    paths = {key: verify_record(value) for key, value in bound_inputs.items()}
    if paths != {key: value.resolve() for key, value in expected_paths.items()}:
        raise ValueError("Input paths differ from the fixed training/development datasets")
    training_inputs = identity["training_provenance"]["inputs"]
    for key, value in training_inputs.items():
        if paths[key] != verify_record(value):
            raise ValueError("Development scorer references different training input files")
    data = FeatureDataset(ROOT, paths["manifest"], paths["features"], paths["metadata"], dimension)
    pool = SourceRetrievalPool(ROOT, paths["development_manifest"], paths["development_features"],
                               paths["development_metadata"], data, dimension)
    for container in (data, pool):
        container.images = F.normalize(container.images, dim=-1).numpy().astype(np.float64)
        container.texts = F.normalize(container.texts, dim=-1).numpy().astype(np.float64)
    if len(data.split_indices["validation"]) != 100 or len(pool.images) != 900 or len(pool.texts) != 4500:
        raise ValueError("Development scope differs from the declared 100/900 image pools")
    return data, pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a development evaluation")
    identity, completion, protocol, model, update_norm = verified_run(args.run, args.protocol_sha256)
    data, pool = load_development(identity, protocol)
    scorer = CanonicalScorer(model.image_mean, model.text_mean, model.image_basis, model.text_basis, model.coefficient)
    output.mkdir(parents=True, exist_ok=True)
    initial_receipt = {"schema": "sanw_v8_development_start_v1", "run": record(root_path(args.run) / "completion.json"),
                       "ledger": record(root_path(args.run) / "ledger.json"), "encoder": identity["encoder"],
                       "seed": identity["config"]["seed"], "protocol": identity["protocol"],
                       "checkpoint": record(root_path(args.run) / completion["selected_checkpoint"]["path"]),
                       "source_sha256": {source: digest(ROOT / source) for source in SOURCES},
                       "inputs": protocol["development_inputs"][identity["encoder"]], "contract": CONTRACT,
                       "checkpoint_selection_precedes_development_read": True,
                       "score": "float64 cosine plus centered rank-r bilinear; unoptimized einsum projections and row-sum pair dots",
                       "feature_preprocessing": "one_Torch_float32_L2_normalization_then_float64",
                       "no_heldout_benchmark_access": True, "alpha_search": False, "checkpoint_search": False}
    start_record = write_json(output / "evaluation_start.json", initial_receipt)
    summaries, artifacts, raws = {}, {}, {}
    for name, state in (("frozen", None), ("trained", scorer)):
        comp, comp_raw = composition_metrics(data, state)
        retrieval, retrieval_raw = exact_retrieval(pool.images, pool.texts, pool.owner.numpy(), state)
        retrieval_raw.update({"image_ids": np.asarray(pool.image_ids), "text_ids": np.asarray(pool.text_ids)})
        summaries[name] = {"composition": comp, "retrieval": retrieval}
        artifacts[f"{name}_composition"] = write_npz(output / f"{name}_composition.npz", comp_raw)
        artifacts[f"{name}_retrieval"] = write_npz(output / f"{name}_retrieval.npz", retrieval_raw)
        raws[name] = {"composition": comp_raw, "retrieval": retrieval_raw}
    retention, composition_effects, samples = {}, {}, {}
    for direction, clusters in (("i2t", np.asarray(pool.image_ids)), ("t2i", np.asarray(pool.image_ids)[pool.owner.numpy()])):
        delta = (raws["trained"]["retrieval"][f"{direction}_correct"].astype(np.float64)
                 - raws["frozen"]["retrieval"][f"{direction}_correct"].astype(np.float64))
        retention[direction], samples[direction] = paired_cluster_bootstrap(delta, clusters)
    for kind in ("original", "source_pair"):
        frozen, trained = raws["frozen"]["composition"], raws["trained"]["composition"]
        if not np.array_equal(frozen[f"{kind}_image_ids"], trained[f"{kind}_image_ids"]):
            raise ValueError("Trained and frozen composition units disagree")
        delta = trained[f"{kind}_joint_accuracy"] - frozen[f"{kind}_joint_accuracy"]
        composition_effects[kind], samples[kind] = paired_cluster_bootstrap(delta, frozen[f"{kind}_image_ids"])
    artifacts["bootstrap_samples"] = write_npz(output / "bootstrap_samples.npz", samples)
    gate = development_gate(summaries["frozen"]["composition"], summaries["trained"]["composition"],
                            retention, update_norm=update_norm)
    result = {"schema": "sanw_v8_development_result_v1", "encoder": identity["encoder"],
              "seed": identity["config"]["seed"], "selected_epoch": completion["selected_epoch"],
              "selected_training_objective": completion["selected_training_objective"],
              "coefficient_frobenius_norm": update_norm, "start_receipt": start_record,
              "summaries": summaries, "retention_effects": retention, "composition_effects": composition_effects,
              "gate": gate, "artifacts": artifacts, "elapsed_seconds": time.monotonic() - started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "interpretation": "development diagnostic only; historical test reuse precludes a fresh confirmatory claim"}
    # Reverify the critical input identities after all predictions are written.
    verified_run(args.run, args.protocol_sha256)
    for value in protocol["development_inputs"][identity["encoder"]].values():
        verify_record(value)
    write_json(output / "result.json", result)
    print(json.dumps({"encoder": identity["encoder"], "seed": identity["config"]["seed"],
                      "gate": gate, "retention": retention, "composition": composition_effects,
                      "result": record(output / "result.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
