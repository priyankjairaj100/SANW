#!/usr/bin/env python3
"""Seal two token-family manifests from six complete canonical dev trajectories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_training import select_development_epoch
from gcr.training import canonical_json, sha256_file


def path(value):
    candidate = Path(value)
    candidate = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError("Path escapes repository")
    return candidate


def read(value):
    return json.loads(path(value).read_text())


def record(value):
    value = path(value)
    return {"path": str(value.relative_to(ROOT)), "sha256": sha256_file(value)}


def verify(item):
    value = path(item["path"])
    if sha256_file(value) != item["sha256"]:
        raise ValueError(f"Bound file changed: {value}")
    return value


def write_new(value, data):
    value = path(value)
    value.parent.mkdir(parents=True, exist_ok=True)
    if value.exists():
        if read(value) != data:
            raise ValueError("Existing selection artifact differs; preserve it")
        return
    with value.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", default="results/practical_v6/canonical_text_development_v1")
    parser.add_argument("--protocol", default="results/practical_v6/token_selection_protocol_v1.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", default="results/practical_v6/token_final_selection")
    args = parser.parse_args()
    protocol_path = verify({"path": args.protocol, "sha256": args.protocol_sha256})
    protocol = read(protocol_path)
    parent = read(verify(protocol["parent_protocol"]))
    if protocol["practical_gate"] != parent["practical_gate"] or protocol["test_lock"] != parent["test_lock"]:
        raise ValueError("Practical gate or test lock changed")
    if protocol["seeds"] != [17, 29, 43] or protocol["required_epochs"] != list(range(5)):
        raise ValueError("Unexpected seed/epoch selection scope")
    for item in protocol["token_training_protocols"]:
        verify(item)
    amendment_path = verify(protocol["canonical_inference_amendment"])
    amendment = read(amendment_path)
    for name, expected in amendment["source_hashes"].items():
        verify({"path": name, "sha256": expected})
    output, canonical_root = path(args.output), path(args.canonical_root)
    all_manifests, final_rows = [], []
    for encoder in protocol["encoders"]:
        completion_path = canonical_root / encoder / "completion.json"
        completion = read(completion_path)
        if completion["encoder"] != encoder or len(completion["snapshots"]) != 3:
            raise ValueError("Encoder canonical completion is incomplete")
        snapshots = []
        states, sources, training_metadata = [], {}, None
        seen_seeds = set()
        for snapshot_record in completion["snapshots"]:
            snapshot = read(verify(snapshot_record))
            seed = snapshot["seed"]
            if seed in seen_seeds or seed not in (17, 29, 43) or snapshot["encoder"] != encoder:
                raise ValueError("Wrong or duplicated seed/encoder")
            seen_seeds.add(seed)
            if not snapshot["trajectory_complete"] or not snapshot["selection_is_finalizable"]:
                raise ValueError("Provisional canonical trajectory")
            rows = snapshot["epochs"]
            if [row["epoch"] for row in rows] != list(range(5)):
                raise ValueError("Missing canonical epoch")
            selected = select_development_epoch(rows)
            if selected != snapshot["selection"] or selected["selected_epoch"] is None:
                raise ValueError("No reproducible eligible nonzero selection")
            binding = read(verify(snapshot["binding"]))
            ledger_path = verify(binding["ledger"])
            ledger = read(ledger_path)
            if hashlib.sha256(canonical_json(ledger["identity"])).hexdigest() != binding["ledger_sha256"]:
                raise ValueError("Training ledger identity mismatch")
            for name, expected in binding["source_sha256"].items():
                verify({"path": name, "sha256": expected})
                if name in sources and sources[name] != expected:
                    raise ValueError("Different canonical source versions across seeds")
                sources[name] = expected
            for name, expected in ledger["identity"]["source_sha256"].items():
                verify({"path": name, "sha256": expected})
                if name in sources and sources[name] != expected:
                    raise ValueError("Training source versions differ")
                sources[name] = expected
            for item in binding["inputs"].values():
                verify(item)
            metadata = binding["inputs"]["metadata"]
            if training_metadata is not None and metadata != training_metadata:
                raise ValueError("Encoder feature provenance differs across seeds")
            training_metadata = metadata
            verify(binding["canonical_prefix_receipt"])
            for row in rows:
                verify(row["checkpoint"])
                verify(row["composition"]["predictions"])
                verify(row["retrieval"]["predictions"])
                verify(row["canonical_text_features"])
            chosen = rows[selected["selected_epoch"]]
            rms = max(chosen["composition"]["residual_rms"], chosen["retrieval"]["residual_rms"])
            if chosen["update_norm"] <= 0 or chosen["optimizer_steps"] <= 0 or rms <= protocol["selection"]["non_negligible_residual_rms_threshold"]:
                raise ValueError("Selected canonical state is effectively unchanged")
            state = {"state_id": f"token_last_block__{encoder}__seed_{seed}__epoch_{chosen['epoch']:02d}",
                     "seed": seed, "epoch": chosen["epoch"], "update_norm": chosen["update_norm"],
                     "optimizer_steps": chosen["optimizer_steps"], "checkpoint": chosen["checkpoint"]["path"],
                     "checkpoint_sha256": chosen["checkpoint"]["sha256"], "canonical_development_snapshot": snapshot_record,
                     "canonical_development_residual_rms": {"composition": chosen["composition"]["residual_rms"], "retrieval": chosen["retrieval"]["residual_rms"]},
                     "development_gains": {name: selected[name] for name in ("paired_joint_accuracy_gain", "i2t_r1_change", "t2i_r1_change")},
                     "training_ledger": record(ledger_path), "canonical_text_features": chosen["canonical_text_features"]}
            states.append(state); snapshots.append(snapshot_record)
            final_rows.append({"encoder": encoder, **state})
        if seen_seeds != {17, 29, 43}:
            raise ValueError("Three selected seeds required")
        sources.update(amendment["source_hashes"])
        sources[str(amendment_path.relative_to(ROOT))] = sha256_file(amendment_path)
        sources[str(protocol_path.relative_to(ROOT))] = args.protocol_sha256
        sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256_file(Path(__file__))
        manifest = {"schema": "sanw_token_evaluation_manifest_v1", "encoder": encoder, "family": "bounded_token_last_block",
                    "protocol_sha256": args.protocol_sha256, "protocol": record(protocol_path),
                    "test_outcomes_used_for_selection": False, "current_test_outcomes_used_for_selection": False,
                    "evaluation_status": "exploratory_after_historical_test_reuse", "training_metadata": training_metadata,
                    "source_hashes": sources, "states": sorted(states, key=lambda state: state["seed"]),
                    "canonical_development_completion": record(completion_path), "canonical_development_snapshots": snapshots,
                    "canonical_inference_amendment": record(amendment_path)}
        manifest_path = output / f"{encoder}_manifest.json"
        write_new(manifest_path, manifest)
        all_manifests.append(record(manifest_path))
    lock = {"schema": "sanw_token_final_selection_lock_v1", "family": "bounded_token_last_block",
            "protocol": record(protocol_path), "manifests": all_manifests, "selected_states": final_rows,
            "test_outcomes_used_for_selection": False, "current_test_outcomes_used_for_selection": False,
            "practical_gate": protocol["practical_gate"], "test_lock": protocol["test_lock"],
            "evaluation_status": "exploratory_after_historical_test_reuse"}
    lock_path = output / "selection_lock.json"
    write_new(lock_path, lock)
    print(json.dumps({"status": "token_selection_locked_before_test", "lock": record(lock_path),
                      "manifests": all_manifests, "states": final_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
