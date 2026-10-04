#!/usr/bin/env python3
"""Verify duplicate baseline executions without treating them as new replicates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from audit_allocation_distillation_independent import Audit, canonical_digest, digest, read_json

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ("vit_b32", "rn50")
POLICIES = ("source", "supported", "allocation_0.5", "allocation_0.8",
            "distilled_1", "distilled_4", "distilled_16")
RATES = (1e-4, 3e-4, 1e-3)
SEEDS = (17, 29, 43)


def tensor_digest(state):
    """Hash sorted parameter names, dtypes, shapes, and raw tensor bytes."""
    h = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().numpy()
        header = json.dumps({"name": name, "dtype": array.dtype.str,
                             "shape": array.shape}, sort_keys=True, separators=(",", ":")).encode()
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def compare_tensors(left, right):
    if set(left) != set(right):
        raise ValueError("Duplicate fits have different parameter names")
    for name in left:
        if left[name].dtype != right[name].dtype or left[name].shape != right[name].shape:
            raise ValueError("Duplicate fits have different tensor types or shapes")
        if not torch.equal(left[name], right[name]):
            difference = float((left[name].double() - right[name].double()).abs().max())
            raise ValueError(f"Duplicate tensor differs: {name}; maximum difference {difference}")
    a, b = tensor_digest(left), tensor_digest(right)
    if a != b:
        raise ValueError("Canonical parameter bytes differ")
    return a


def inspect_candidate(audit, directory, ledger, policy, rate, seed, *, replication=False):
    completion = read_json(directory / "completion.json")
    expected = {"ledger_sha256": ledger["ledger_sha256"], "method": policy,
                "learning_rate": rate, "seed": seed, "epochs_completed": 10,
                "checkpoint_digest_count" if replication else "checkpoint_count": 11}
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError("A duplicate baseline candidate is incomplete or has different provenance")
    for name in ("history.json", "epochs/epoch_10.pt"):
        audit.check_hash(directory / name, completion["artifact_sha256"][name])
    document = read_json(directory / "history.json")
    if any(document.get(key) != expected[key] for key in ("ledger_sha256", "method", "learning_rate", "seed")):
        raise ValueError("Duplicate baseline history provenance differs")
    if [row["epoch"] for row in document["history"]] != list(range(11)):
        raise ValueError("Duplicate baseline histories must contain all epochs")
    if replication:
        checkpoint_hashes = completion.get("checkpoint_sha256", {})
        if set(checkpoint_hashes) != {f"epochs/epoch_{epoch:02d}.pt" for epoch in range(11)}:
            raise ValueError("Replication requires all eleven checkpoint digests")
        if "epochs/epoch_10.pt" not in completion.get("retained_checkpoints", []):
            raise ValueError("Replication must retain the terminal checkpoint")
        for row in document["history"]:
            name = f"epochs/epoch_{row['epoch']:02d}.pt"
            if row["checkpoint"] != name or row["checkpoint_sha256"] != checkpoint_hashes[name]:
                raise ValueError("Replication history checkpoint binding differs")
        if checkpoint_hashes["epochs/epoch_10.pt"] != completion["artifact_sha256"]["epochs/epoch_10.pt"]:
            raise ValueError("Replication terminal checkpoint binding differs")
    checkpoint = torch.load(directory / "epochs/epoch_10.pt", map_location="cpu", weights_only=True)
    for key in ("ledger_sha256", "method", "learning_rate", "seed"):
        if checkpoint[key] != expected[key]:
            raise ValueError("Duplicate checkpoint metadata differs from its completion")
    if checkpoint["epoch"] != 10 or checkpoint["protocol_sha256"] != ledger["identity"]["protocol"]["sha256"]:
        raise ValueError("Duplicate checkpoint uses a different protocol or epoch")
    return document["history"], checkpoint


def compare_replication_history(retention_history, replication_history):
    """Compare all common metrics at alpha=1, without dropping metric fields."""
    if len(retention_history) != 11 or len(replication_history) != 11:
        raise ValueError("Duplicate baseline histories must contain all epochs")
    for epoch, (left, right) in enumerate(zip(retention_history, replication_history, strict=True)):
        if left["epoch"] != epoch or right["epoch"] != epoch:
            raise ValueError("Duplicate baseline epoch ordering differs")
        for field in ("mean_training_loss", "training_images", "gradient_steps"):
            if left[field] != right[field]:
                raise ValueError(f"Replication duplicate trajectory differs: epoch {epoch} {field}")
        unscaled = [row for row in left["validation"] if row["alpha"] == 1.]
        if len(unscaled) != 1:
            raise ValueError("Retention validation requires one unscaled state")
        if set(right["validation"]) != {"native", "source_retrieval"}:
            raise ValueError("Unexpected replication validation schema")
        for field in ("native", "source_retrieval"):
            # Replication also saves native item rows; retention saves their summary.
            reference = {key: value for key, value in right["validation"][field].items()
                         if key != "per_image"}
            if unscaled[0][field] != reference:
                raise ValueError(f"Replication duplicate validation differs: epoch {epoch} {field}")


def compare_replication(audit, ad, replication_path):
    """Check 18 RN50 Source/U repeats against the already verified AD baselines."""
    replication_path = audit.path(replication_path)
    ledger = read_json(replication_path / "ledger.json")
    identity = ledger["identity"]
    if canonical_digest(identity) != ledger["ledger_sha256"] or identity.get("setting") != "rn50":
        raise ValueError("Invalid RN50 replication ledger")
    for name, expected in identity["source_sha256"].items():
        audit.check_hash(name, expected)
    audit.check_hash(identity["protocol"]["path"], identity["protocol"]["sha256"])
    ad_path, ad_ledger = ad["rn50"]
    if (ad_ledger["identity"]["architecture"] != "linear" or
            identity["adapter"].get("kind") != "linear" or
            identity["adapter"].get("dimension") != ad_ledger["identity"]["hyperparameters"]["feature_dim"]):
        raise ValueError("Replication duplicate adapter differs")
    for left_field, right_field in (("hyperparameters", "base_hyperparameters"),
                                    ("inputs", "training_inputs"),
                                    ("development_retrieval", "source_retrieval_validation")):
        if ad_ledger["identity"][left_field] != identity[right_field]:
            raise ValueError(f"Replication duplicate fits used different {left_field}")
    records = []
    for policy in ("source", "supported"):
        for rate in RATES:
            for seed in SEEDS:
                relative = Path("candidates") / policy / f"lr_{rate:.8g}" / f"seed_{seed}"
                left_history, left = inspect_candidate(audit, ad_path / relative, ad_ledger, policy, rate, seed)
                right_history, right = inspect_candidate(audit, replication_path / relative, ledger,
                                                         policy, rate, seed, replication=True)
                compare_replication_history(left_history, right_history)
                parameter_hash = compare_tensors(left["state_dict"], right["state_dict"])
                records.append({"encoder": "rn50", "policy": policy, "learning_rate": rate, "seed": seed,
                    "terminal_epoch": 10, "canonical_parameters_sha256": parameter_hash,
                    "ad_checkpoint": str((ad_path / relative / "epochs/epoch_10.pt").relative_to(ROOT)),
                    "replication_checkpoint": str((replication_path / relative / "epochs/epoch_10.pt").relative_to(ROOT)),
                    "all_eleven_common_training_and_development_rows_identical": True})
    return records, replication_path / "ledger.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-outputs", nargs=2, required=True)
    parser.add_argument("--retention-outputs", nargs=2, required=True)
    parser.add_argument("--rn50-replication-output",
                        help="Optionally audit 18 additional RN50 Source/U duplicate executions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = Audit(ROOT)
    def outputs(paths):
        result = {}
        for value in paths:
            path = audit.path(value)
            ledger = read_json(path / "ledger.json")
            if canonical_digest(ledger["identity"]) != ledger["ledger_sha256"]:
                raise ValueError("Execution ledger identity is invalid")
            encoder = ledger["identity"]["encoder"]
            if encoder in result or encoder not in ENCODERS:
                raise ValueError("Duplicate or unknown encoder")
            for name, expected in ledger["identity"]["source_sha256"].items():
                audit.check_hash(name, expected)
            protocol = ledger["identity"]["protocol"]
            audit.check_hash(protocol["path"], protocol["sha256"])
            result[encoder] = path, ledger
        if set(result) != set(ENCODERS):
            raise ValueError("Both encoder outputs are required")
        return result
    ad, retention = outputs(args.ad_outputs), outputs(args.retention_outputs)
    records = []
    for encoder in ENCODERS:
        left_root, left_ledger = ad[encoder]
        right_root, right_ledger = retention[encoder]
        for field in ("hyperparameters", "inputs", "development_retrieval"):
            if left_ledger["identity"][field] != right_ledger["identity"][field]:
                raise ValueError(f"Duplicate fits used different {field}")
        for policy in POLICIES:
            for rate in RATES:
                for seed in SEEDS:
                    relative = Path("candidates") / policy / f"lr_{rate:.8g}" / f"seed_{seed}"
                    left_history, left = inspect_candidate(audit, left_root / relative, left_ledger, policy, rate, seed)
                    right_history, right = inspect_candidate(audit, right_root / relative, right_ledger, policy, rate, seed)
                    for a, b in zip(left_history, right_history, strict=True):
                        for field in ("epoch", "mean_training_loss", "training_images", "gradient_steps", "validation"):
                            if a[field] != b[field]:
                                raise ValueError(f"Duplicate trajectory differs: {encoder} {policy} {rate} {seed} {field}")
                    parameter_hash = compare_tensors(left["state_dict"], right["state_dict"])
                    records.append({"encoder": encoder, "policy": policy, "learning_rate": rate, "seed": seed,
                        "terminal_epoch": 10, "canonical_parameters_sha256": parameter_hash,
                        "ad_checkpoint": str((left_root / relative / "epochs/epoch_10.pt").relative_to(ROOT)),
                        "retention_checkpoint": str((right_root / relative / "epochs/epoch_10.pt").relative_to(ROOT)),
                        "all_eleven_training_and_development_rows_identical": True})
        print(json.dumps({"encoder": encoder, "identical_terminal_fits": 63}), flush=True)
    replication_records, replication_ledger_path = ([], None)
    if args.rn50_replication_output:
        replication_records, replication_ledger_path = compare_replication(audit, ad, args.rn50_replication_output)
    result = {"schema_version": 2, "status": "passed", "duplicate_terminal_fits": len(records),
        "identical_trajectory_rows": 11 * len(records), "records": records,
        "rn50_replication_duplicate_terminal_fits": len(replication_records),
        "rn50_replication_identical_common_trajectory_rows": 11 * len(replication_records),
        "rn50_replication_records": replication_records,
        "total_pairwise_duplicate_comparisons": len(records) + len(replication_records),
        "interpretation": "These are deterministic duplicate executions of the same three seeds, not additional independent replicates.",
        "scope": "exact terminal tensor bytes and all eleven training/development summary rows per common baseline",
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__),
            "scripts/audit_allocation_distillation_independent.py": digest(ROOT / "scripts/audit_allocation_distillation_independent.py")},
        "ledger_file_sha256": {str((path / "ledger.json").relative_to(ROOT)): digest(path / "ledger.json")
                               for mapping in (ad, retention) for path, _ in mapping.values()},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)}}
    if replication_ledger_path is not None:
        result["ledger_file_sha256"][str(replication_ledger_path.relative_to(ROOT))] = digest(replication_ledger_path)
    output = audit.path(args.output)
    if output.exists() and read_json(output) != result:
        raise ValueError("Existing duplicate execution audit differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "duplicate_terminal_fits": len(records),
                     "rn50_replication_duplicate_terminal_fits": len(replication_records),
                     "total_pairwise_duplicate_comparisons": len(records) + len(replication_records)}), flush=True)


if __name__ == "__main__":
    main()
