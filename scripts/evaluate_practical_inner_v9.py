#!/usr/bin/env python3
"""Evaluate completed, protocol-bound inner fits on the common v9 holdout.

No fitting, official-development evaluation, or benchmark loading occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_practical_constrained_development_v8 import digest, read, record, root_path, verify_record, write_json, write_npz
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer
from gcr.practical_inner_evaluation_v9 import LABCLIPScorer, evaluate_inner, exact_changes, make_inner_view, paired_changes
from gcr.training import FeatureDataset


EVALUATION_SOURCES = ("scripts/evaluate_practical_inner_v9.py", "src/gcr/practical_inner_evaluation_v9.py",
                      "src/gcr/practical_constrained_evaluation_v8.py", "src/gcr/training.py",
                      "scripts/evaluate_practical_constrained_development_v8.py")


def verified_inner_run(run, protocol_hash):
    run = root_path(run)
    ledger, completion = read(run / "ledger.json"), read(run / "completion.json")
    identity = ledger["identity"]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(payload).hexdigest() != ledger["ledger_sha256"] or completion["ledger_sha256"] != ledger["ledger_sha256"]:
        raise ValueError("Inner fit ledger hash mismatch")
    protocol_path = verify_record(identity["protocol"])
    if digest(protocol_path) != protocol_hash:
        raise ValueError("Protocol hash mismatch")
    protocol = read(protocol_path)
    if protocol["study"] != "sanw_practical_v9" or identity["mode"] != "inner" or completion["mode"] != "inner":
        raise ValueError("Only original-training inner fits may be evaluated here")
    if identity["encoder"] not in protocol["encoders"] or completion["encoder"] != identity["encoder"]:
        raise ValueError("Inner encoder mismatch")
    if not set(EVALUATION_SOURCES) <= set(protocol["source_sha256"]):
        raise ValueError("Protocol does not bind the complete inner evaluator")
    for source, expected in protocol["source_sha256"].items():
        if digest(ROOT / source) != expected:
            raise ValueError(f"Protocol-bound source changed: {source}")
    for source, expected in identity["source_sha256"].items():
        if expected != protocol["source_sha256"][source]:
            raise ValueError("Fit source differs from protocol")
    if identity["split"] != protocol["split"]:
        raise ValueError("Fit split differs from the common owner split")
    split = read(verify_record(protocol["split"]))
    inputs = identity["training_provenance"]["inputs"]
    if inputs != protocol["training_inputs"][identity["encoder"]] or split["manifest"] != inputs["manifest"]:
        raise ValueError("Fit input identity differs from the declared original-training archive")
    for value in inputs.values():
        verify_record(value)
    if identity["config"]["seed"] != 17 or completion["config"] != identity["config"]:
        raise ValueError("Inner fit seed/config mismatch")
    states = []
    config = identity["config"]
    if identity["study"] == "sanw_practical_v9" and identity["family"] == "joint":
        expected = {**protocol["joint"]["base_config"], "radius": config["radius"],
                    "composition_weight": config["composition_weight"], "seed": 17}
        if config != expected or config["radius"] not in protocol["joint"]["radii"] or config["composition_weight"] not in protocol["joint"]["composition_weights"]:
            raise ValueError("Joint state falls outside the frozen finite grid")
        if identity["training_provenance"]["training_image_manifest_indices"] != sorted(split["train_image_manifest_indices"]):
            raise ValueError("Joint fitting used owners outside the 960-image fit partition")
        history = completion["history"]
        if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError("Incomplete fixed-budget joint trajectory")
        if any(not np.isfinite(row["training_objective"]) for row in history):
            raise ValueError("Nonfinite joint training selection objective")
        eligible = [row for row in history if row["nonzero"]]
        if not eligible:
            raise ValueError("No nonzero joint state")
        chosen = min(eligible, key=lambda row: (row["training_objective"], row["epoch"]))
        if (chosen["epoch"] != completion["selected_epoch"]
                or chosen["training_objective"] != completion["selected_training_objective"]
                or completion["selection"] != "minimum_feasible_nonzero_training_objective_then_earliest_epoch"):
            raise ValueError("Joint checkpoint selection differs from the training-only rule")
        certificate = completion["final_certificate"]
        if not all(certificate.get(key) is True for key in ("ranking_preserved", "feasible_with_tolerance", "ranking_checked_canonically")):
            raise ValueError("Joint state lacks its canonical training feasibility certificate")
        path = run / completion["selected_checkpoint"]["path"]
        if digest(path) != completion["selected_checkpoint"]["sha256"]:
            raise ValueError("Selected joint checkpoint changed")
        model = ConstrainedBilinearScorer.load(path)
        epoch = completion["checkpoint_history"][chosen["epoch"] - 1]
        epoch_path = run / epoch["checkpoint"]["path"]
        if digest(epoch_path) != epoch["checkpoint"]["sha256"]:
            raise ValueError("Training-selected epoch checkpoint changed")
        saved = ConstrainedBilinearScorer.load(epoch_path)
        for key in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient"):
            if not np.array_equal(getattr(model, key), getattr(saved, key)):
                raise ValueError("Joint selected state differs from its selected training epoch")
        norm = float(np.linalg.norm(model.coefficient))
        if not 0 < norm <= config["radius"] * (1 + 64 * np.finfo(np.float64).eps):
            raise ValueError("Joint coefficient is zero or exceeds its declared radius")
        scorer = CanonicalScorer(model.image_mean, model.text_mean, model.image_basis, model.text_basis, model.coefficient)
        states.append({"family": "joint", "epoch": chosen["epoch"], "radius": config["radius"],
                       "composition_weight": config["composition_weight"], "checkpoint": record(path),
                       "nonzero_functional_update": True, "scorer": scorer})
    elif identity["study"] == "sanw_labclip_inner_v9":
        arm = config["arm"]
        expected = {**protocol["labclip"]["arms"][arm]["config"], "seed": 17}
        if config != expected or completion["train_only"] is not True or completion["official_development_or_test_read"] is not False:
            raise ValueError("LABCLIP fit differs from the declared training-only recipe")
        global_rows = identity["training_provenance"]["training_image_manifest_indices"]
        requested_local = identity["requested_fit_loader_owner_indices"]
        used_local = identity["fit_loader_owner_indices"]
        omitted_local = identity["omitted_no_valid_contradiction_loader_owner_indices"]
        if (len(set(requested_local)) != len(requested_local) or len(set(used_local)) != len(used_local)
                or len(set(omitted_local)) != len(omitted_local) or set(used_local) & set(omitted_local)
                or set(used_local) | set(omitted_local) != set(requested_local)):
            raise ValueError("LABCLIP used/omitted owners do not partition its requested fit owners")
        if sorted(global_rows[i] for i in requested_local) != sorted(split["train_image_manifest_indices"]):
            raise ValueError("LABCLIP requested fitting used inner holdout owners")
        # Independently establish every omission from the stated conflict rule.
        manifest = read(inputs["manifest"]["path"])
        text_values = {str(row["id"]): tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", row["text"]).casefold()))
                       for row in manifest["texts"]}
        requested_ids = {str(manifest["images"][global_rows[i]]["id"]): i for i in requested_local}
        by_owner = {key: [] for key in requested_ids}
        for pair in manifest["pairs"]:
            if str(pair["image_id"]) in by_owner:
                by_owner[str(pair["image_id"])].append(pair)
        expected_omissions = []
        for image_id, pairs in by_owner.items():
            positives = {text_values[str(pair["text_id"])] for pair in pairs if pair["relation"] in ("source", "supported")}
            negatives = [pair for pair in pairs if pair["relation"] == "contradicted" and text_values[str(pair["text_id"])] not in positives]
            if not negatives:
                expected_omissions.append(requested_ids[image_id])
        if sorted(omitted_local) != sorted(expected_omissions):
            raise ValueError("LABCLIP omissions differ from independently reconstructed negative eligibility")
        if [row["epoch"] for row in completion["history"]] != list(range(1, config["epochs"] + 1)):
            raise ValueError("Incomplete LABCLIP fixed training budget")
        epochs = [0] + config["checkpoint_epochs"]
        if [row["epoch"] for row in completion["checkpoint_history"]] != epochs:
            raise ValueError("LABCLIP saved epoch coverage differs from protocol")
        for row in completion["checkpoint_history"]:
            path = run / row["checkpoint"]["path"]
            if digest(path) != row["checkpoint"]["sha256"]:
                raise ValueError("LABCLIP checkpoint changed")
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["schema"]) != "sanw_labclip_inner_v9":
                    raise ValueError("Wrong LABCLIP state schema")
                weight = archive["weight"]
            if row["epoch"] == 0 and not np.array_equal(weight, np.eye(len(weight), dtype=weight.dtype)):
                raise ValueError("LABCLIP initial state is not exact identity")
            states.append({"family": "labclip" if arm == "small_data_fixed_scale" else "labclip_published_control",
                           "arm": arm, "epoch": row["epoch"], "checkpoint": record(path),
                           "nonzero_functional_update": row["nonzero_functional_update"], "scorer": LABCLIPScorer(weight)})
    else:
        raise ValueError("Unrecognized inner fitting family")
    return identity, completion, protocol, split, states


def load_inner_view(identity, split):
    paths = {name: verify_record(value) for name, value in identity["training_provenance"]["inputs"].items()}
    data = FeatureDataset(ROOT, paths["manifest"], paths["features"], paths["metadata"],
                          512 if identity["encoder"] == "vit_b32" else 1024)
    data.images = F.normalize(data.images, dim=-1).numpy().astype(np.float64)
    data.texts = F.normalize(data.texts, dim=-1).numpy().astype(np.float64)
    return make_inner_view(data, split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = root_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite inner evaluation")
    identity, completion, protocol, split, states = verified_inner_run(args.run, args.protocol_sha256)
    view = load_inner_view(identity, split)
    output.mkdir(parents=True, exist_ok=True)
    start = {"study": "sanw_practical_v9_inner_evaluation", "protocol": identity["protocol"], "split": protocol["split"],
             "ledger": record(root_path(args.run) / "ledger.json"), "completion": record(root_path(args.run) / "completion.json"),
             "encoder": identity["encoder"], "inputs": identity["training_provenance"]["inputs"],
             "sources": {name: digest(ROOT / name) for name in EVALUATION_SOURCES},
             "official_development_or_benchmarks_used": False}
    start_record = write_json(output / "evaluation_start.json", start)
    frozen_summary, frozen = evaluate_inner(view)
    frozen_record = write_npz(output / "frozen.npz", frozen)
    identity_raw, identity_record = None, None
    results = []
    for state in states:
        summary, raw = evaluate_inner(view, state["scorer"])
        prediction_record = write_npz(output / f"epoch_{state['epoch']:03d}.npz", raw)
        changes, paired = paired_changes(frozen, raw)
        paired_record = write_npz(output / f"epoch_{state['epoch']:03d}_paired.npz", paired)
        result = {key: value for key, value in state.items() if key != "scorer"}
        result.update({"study": "sanw_practical_v9_inner_state", "encoder": identity["encoder"], "seed": 17,
                       "protocol_sha256": args.protocol_sha256, "split": protocol["split"], "start_receipt": start_record,
                       "frozen": frozen_record, "predictions": prediction_record, "paired_predictions": paired_record,
                       "frozen_summary": frozen_summary, "summary": summary, "paired_changes": changes,
                       "exact_paired_changes": exact_changes(paired), "official_development_or_benchmarks_used": False})
        if state["family"].startswith("labclip"):
            if state["epoch"] == 0:
                identity_raw, identity_record = raw, prediction_record
                result["identity_control"] = True
            else:
                if identity_raw is None:
                    raise ValueError("LABCLIP identity must be evaluated before learned states")
                identity_changes, identity_paired = paired_changes(identity_raw, raw)
                result.update({"identity": identity_record, "identity_paired_changes": identity_changes,
                               "exact_identity_paired_changes": exact_changes(identity_paired),
                               "identity_paired_predictions": write_npz(output / f"epoch_{state['epoch']:03d}_vs_identity.npz", identity_paired)})
        result_record = write_json(output / f"epoch_{state['epoch']:03d}.json", result)
        results.append({"epoch": state["epoch"], "family": state["family"], "result": result_record})
        print(json.dumps({"encoder": identity["encoder"], "family": state["family"], "epoch": state["epoch"],
                          "paired_changes": changes, "learned_vs_identity": result.get("identity_paired_changes")}), flush=True)
    verified_inner_run(args.run, args.protocol_sha256)
    index = {"study": "sanw_practical_v9_inner_index", "encoder": identity["encoder"], "start_receipt": start_record,
             "protocol_sha256": args.protocol_sha256, "states": results}
    write_json(output / "index.json", index)


if __name__ == "__main__":
    main()
