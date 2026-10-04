#!/usr/bin/env python3
"""Score complete, frozen replication plans after development selection.

This new execution preserves the original replication code and protocols.
It shares the allocation/distillation dataset and scoring backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from gcr.evaluation import PAIR_TIE_POLICY, RETRIEVAL_TIE_POLICY
from gcr.review_training import candidate_path, file_record, state_id, _atomic_npz
from gcr.strengthen_replication import (POLICIES, SELECTORS, _completed,
    load_replication_adapter)
from gcr.training import canonical_json, atomic_json, sha256_file

PROTOCOL_SHA256 = {
    "nonlinear": "3d1d651ebe9ddfd7a1bd71f403eb0497a46ebb16227c25e206bf21e92431d920",
    "rn50": "53a8df510903144faf995b6935bd1049c804664d3e794a7e2ab9589b884de85e",
}
SEEDS = (17, 29, 43)
RATES = (0.0001, 0.0003, 0.001)
DATASETS = ("e_vil_test1000", "visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
EVIDENCE_TYPE = "strengthening_replication_fresh_execution_20261004"


def at_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_protocol(path: Path, setting: str, expected: str) -> dict:
    if expected != PROTOCOL_SHA256[setting] or sha256_file(path) != expected:
        raise ValueError("Protocol must match the original frozen replication digest")
    protocol = json.loads(path.read_text())
    primary = protocol["primary_analysis"]
    expected_primary = {"epoch": 10, "family_size": 12, "alpha": .05,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261013 if setting == "nonlinear" else 20261014,
        "dataset": "e_vil_test1000", "endpoints": ["i2t.r1", "t2i.r1"],
        "learning_rates": list(RATES),
        "contrasts": ["supported-minus-source", "supported-minus-score-stratified"]}
    if any(primary.get(key) != value for key, value in expected_primary.items()):
        raise ValueError("Unexpected frozen primary analysis")
    if protocol["evaluation"]["datasets"] != list(DATASETS):
        raise ValueError("All five frozen evaluation datasets are required")
    return protocol


def check_record(record: dict) -> Path:
    path = at_root(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Bound file changed: {record['path']}")
    return path


def verify_selection_plan(manifest: dict, full: dict) -> dict:
    """Recompute every development selection from all 495 epoch records."""
    states = full["states"]
    expected = {(method, rate, seed, epoch) for method in POLICIES
                for rate in RATES for seed in SEEDS for epoch in range(11)}
    keys = [(s["method"], s["learning_rate"], s["seed"], s["epoch"]) for s in states]
    if len(states) != 495 or len(set(keys)) != 495 or set(keys) != expected:
        raise ValueError("Full manifest must contain the exact 495-record grid")
    by_key = dict(zip(keys, states))
    expected_selections = []
    for selector in SELECTORS:
        for method in POLICIES:
            candidates = []
            for rate in RATES:
                best = {}
                for seed in SEEDS:
                    rows = [by_key[(method, rate, seed, epoch)] for epoch in range(11)]
                    scores = [row["validation"][selector]["score"] for row in rows]
                    if not np.isfinite(scores).all():
                        raise ValueError("Development scores must be finite")
                    best[seed] = min(rows, key=lambda row: (-row["validation"][selector]["score"], row["epoch"]))
                score = float(np.mean([row["validation"][selector]["score"] for row in best.values()]))
                candidates.append((score, rate, best))
            _, rate, best = min(candidates, key=lambda row: (-row[0], row[1]))
            for seed in SEEDS:
                row = best[seed]
                candidate_id = state_id(method, rate, seed, row["epoch"])
                expected_selections.append({"selector": selector, "method": method,
                    "learning_rate": rate, "seed": seed, "epoch": row["epoch"],
                    "candidate_state_id": candidate_id,
                    "state_id": candidate_id if row["epoch"] else "frozen"})
    selections = manifest["selections"]
    if len(selections) != 30:
        raise ValueError("Replication requires thirty selected seed states")
    lookup = {(s["selector"], s["method"], s["seed"]): s for s in selections}
    if len(lookup) != 30:
        raise ValueError("Duplicate selected seed state")
    for wanted in expected_selections:
        actual = lookup[(wanted["selector"], wanted["method"], wanted["seed"])]
        if any(actual.get(key) != value for key, value in wanted.items()):
            raise ValueError(f"Selection differs from development histories: {wanted}")
    selected_ids = {s["state_id"] for s in expected_selections if s["epoch"]}
    desired = {s["state_id"] for s in states if s["epoch"] == 10} | selected_ids | {"frozen"}
    evaluation = manifest["states"]
    if len(evaluation) != len(desired) or {s["state_id"] for s in evaluation} != desired:
        raise ValueError("Evaluation plan differs from frozen plus terminal and selected states")
    original = {s["state_id"]: s for s in states}
    for row in evaluation:
        if row["state_id"] == "frozen":
            if row["checkpoint"] is not None or row["epoch"] != 0:
                raise ValueError("Invalid frozen reference")
        elif row != original[row["state_id"]] or not row["checkpoint_retained"]:
            raise ValueError("Evaluation state differs from its retained source state")
    return {"status": "passed", "grid_records": 495, "selected_seed_states": 30,
            "evaluation_states": len(evaluation), "test_outcomes_used": False}


def load_and_verify_plan(manifest_path: Path, setting: str, protocol_hash: str):
    manifest = json.loads(manifest_path.read_text())
    ledger_path = manifest_path.parent / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    digest = hashlib.sha256(canonical_json(ledger["identity"])).hexdigest()
    if digest != ledger["ledger_sha256"] or digest != manifest["ledger_sha256"]:
        raise ValueError("Replication ledger identity mismatch")
    for document in (manifest,):
        if document["setting"] != setting or document["protocol_sha256"] != protocol_hash:
            raise ValueError("Replication plan setting or protocol mismatch")
        if document["test_outcomes_used_for_selection"] is not False:
            raise ValueError("Evaluation requires development-only selection")
    identity = ledger["identity"]
    if identity["setting"] != setting or identity["protocol"]["sha256"] != protocol_hash:
        raise ValueError("Ledger setting or protocol mismatch")
    for name, digest in identity["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Frozen fitting implementation changed: {name}")
    for group in ("training_inputs", "source_retrieval_validation"):
        for name in ("manifest", "features", "metadata"):
            check_record(identity[group][name])
    check_record(identity["assignment_receipt"])
    for record in identity["assignment_files"].values():
        check_record(record)
    full_path = check_record(manifest["full_state_manifest"])
    full = json.loads(full_path.read_text())
    for name in ("setting", "ledger_sha256", "protocol_sha256", "adapter"):
        if full[name] != manifest[name]:
            raise ValueError("Full and evaluation manifests disagree")
    audit = verify_selection_plan(manifest, full)
    completion_records = []
    for method in POLICIES:
        for rate in RATES:
            for seed in SEEDS:
                directory = candidate_path(manifest_path.parent, method, rate, seed)
                if _completed(directory, ledger, method, rate, seed) is None:
                    raise ValueError("Every candidate requires a complete artifact receipt")
                completion_records.append(file_record(ROOT, directory / "completion.json"))
    for state in manifest["states"]:
        if state["checkpoint"] is not None:
            if sha256_file(ROOT / state["checkpoint"]) != state["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint digest mismatch")
    audit["candidate_completion_receipts"] = completion_records
    return manifest, ledger, ledger_path, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=PROTOCOL_SHA256, required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    for name in ("manifest", "protocol", "dataset_config", "output"):
        setattr(args, name, at_root(getattr(args, name)))
    validate_protocol(args.protocol, args.setting, args.protocol_sha256)
    manifest, ledger, ledger_path, selection_audit = load_and_verify_plan(args.manifest, args.setting, args.protocol_sha256)
    if args.block_size < 1 or args.torch_threads != 2:
        raise ValueError("Use a positive retrieval block and the declared two CPU threads")
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    from evaluate_allocation_distillation import load_dataset_config, score_state, SOURCE_PATHS
    encoder = "vit_b32" if args.setting == "nonlinear" else "rn50"
    # The shared backend reads AD's common input/provenance layout.
    backend_ledger = {**ledger, "identity": {**ledger["identity"],
        "inputs": ledger["identity"]["training_inputs"], "encoder": encoder}}
    datasets = load_dataset_config(ROOT, args.dataset_config, encoder, backend_ledger)
    if set(datasets) != set(DATASETS):
        raise ValueError("The complete five-dataset plan is mandatory")
    source_names = ("scripts/evaluate_strengthen_replication.py", "scripts/evaluate_allocation_distillation.py",
        "scripts/evaluate_review_followup.py", "scripts/evaluate_study.py", "src/gcr/evaluation.py",
        "src/gcr/adapters.py", "src/gcr/strengthen_replication.py", "src/gcr/review_training.py",
        "src/gcr/training.py")
    source_hashes = {name: sha256_file(ROOT / name) for name in sorted(set(source_names) | set(SOURCE_PATHS))}
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "torch_threads": args.torch_threads, "blas_thread_environment":
        {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    receipt = {"schema_version": 1, "setting": args.setting, "encoder": encoder,
        "evidence_type": EVIDENCE_TYPE, "protocol_sha256": args.protocol_sha256,
        "state_manifest": file_record(ROOT, args.manifest), "ledger": file_record(ROOT, ledger_path),
        "dataset_config": file_record(ROOT, args.dataset_config), "source_hashes": source_hashes,
        "input_hashes": {name: datasets[name]["hashes"] for name in DATASETS},
        "states": manifest["states"], "selections": manifest["selections"],
        "selection_audit": selection_audit, "environment": environment,
        "block_size": args.block_size, "reused_test_sets": True,
        "previous_missing_outputs_used": False}
    if args.plan_only:
        print(json.dumps({"status": "plan_verified_without_scoring", "setting": args.setting,
            "states": manifest["state_count"], "datasets": list(DATASETS), "selection_audit": selection_audit}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "prescore_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Pre-score receipt changed; use a new execution output")
    atomic_json(receipt_path, receipt)
    receipt_hash = sha256_file(receipt_path)
    index = {"schema_version": 1, "status": "running", "setting": args.setting,
        "encoder": encoder, "evidence_type": EVIDENCE_TYPE,
        "protocol_sha256": args.protocol_sha256, "ledger_sha256": ledger["ledger_sha256"],
        "prescore_receipt": str(receipt_path.relative_to(ROOT)), "prescore_receipt_sha256": receipt_hash,
        "datasets": list(DATASETS), "selections": manifest["selections"], "runs": [],
        "retrieval_tie_policy": RETRIEVAL_TIE_POLICY, "pair_tie_policy": PAIR_TIE_POLICY,
        "normalization": "float32 L2 for both adapted and frozen features",
        "score_precision": "float64 unscaled similarities"}
    for state in manifest["states"]:
        adapter = None
        if state["checkpoint"] is not None:
            payload = torch.load(ROOT / state["checkpoint"], map_location="cpu", weights_only=True)
            for key in ("state_id", "method", "condition", "draw_id", "assignment_seed", "learning_rate", "seed", "epoch", "adapter"):
                if payload[key] != state[key]:
                    raise ValueError(f"Checkpoint metadata mismatch: {key}")
            if payload["protocol_sha256"] != args.protocol_sha256 or payload["ledger_sha256"] != ledger["ledger_sha256"]:
                raise ValueError("Checkpoint protocol or ledger binding mismatch")
            adapter = load_replication_adapter(payload)
        row = {**state, "datasets": {}}
        for name in DATASETS:
            started = time.monotonic()
            folder = args.output / "predictions" / state["state_id"]
            folder.mkdir(parents=True, exist_ok=True)
            raw_path, metadata_path = folder / f"{name}.npz", folder / f"{name}.json"
            provenance = {"prescore_receipt_sha256": receipt_hash, "state": state,
                "dataset": name, "input_hashes": datasets[name]["hashes"]}
            reused = False
            if raw_path.exists() or metadata_path.exists():
                if not raw_path.exists() or not metadata_path.exists():
                    raise ValueError("Incomplete cached prediction pair requires inspection")
                saved = json.loads(metadata_path.read_text())
                if saved["provenance"] != provenance or saved["predictions_sha256"] != sha256_file(raw_path):
                    raise ValueError("Cached predictions have different provenance")
                metrics, reused = saved["metrics"], True
            else:
                metrics, predictions = score_state(adapter, datasets[name], args.block_size)
                _atomic_npz(raw_path, **predictions)
                atomic_json(metadata_path, {"metrics": metrics, "provenance": provenance,
                    "predictions_sha256": sha256_file(raw_path), "seconds": time.monotonic() - started})
            row["datasets"][name] = {"metrics": metrics, "predictions": str(raw_path.relative_to(ROOT)),
                "metadata": str(metadata_path.relative_to(ROOT)), "predictions_sha256": sha256_file(raw_path),
                "metadata_sha256": sha256_file(metadata_path)}
            print(json.dumps({"state": state["state_id"], "dataset": name, "reused": reused,
                "seconds": round(time.monotonic() - started, 3)}), flush=True)
        index["runs"].append(row)
        atomic_json(args.output / "index.json", index)
    index.update(status="complete", run_count=len(index["runs"]))
    atomic_json(args.output / "index.json", index)


if __name__ == "__main__":
    main()
