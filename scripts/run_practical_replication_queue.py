#!/usr/bin/env python3
"""Replicate the selected common development configuration on seeds29 and43."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 3:
        raise ValueError("Replication queue permits at most three threads.")
    repository = args.repository.resolve()
    plan_path = repository / "results/practical_v6/common_configuration_replication_plan.json"
    if sha(plan_path) != args.plan_sha256:
        raise ValueError("Replication plan changed.")
    plan = json.loads(plan_path.read_text())
    if plan["epsilon"] != 0.01 or plan["retention_weight"] != 0.25 or plan["additional_seeds"] != [29, 43]:
        raise ValueError("Unexpected replication design.")
    root = repository / "results/practical_v6" / args.encoder
    receipt_path = root / "common_configuration_replication_queue.json"
    if receipt_path.exists():
        raise FileExistsError("Replication queue already has a receipt; inspect before restarting.")
    receipt = {"schema": "sanw_practical_replication_queue_v1", "encoder": args.encoder,
               "plan_sha256": args.plan_sha256, "created_utc": timestamp(), "phase": "ready", "completed": []}
    write_json(receipt_path, receipt)
    for seed in (29, 43):
        if sha(plan_path) != args.plan_sha256:
            raise ValueError("Replication plan changed during execution.")
        for name, expected in plan["source_sha256"].items():
            if sha(repository / name) != expected:
                raise ValueError(f"Frozen fit source changed: {name}")
        name = f"eps_0.01_ret_0.25_seed_{seed}"
        output = root / name
        if output.exists():
            raise FileExistsError(f"Candidate already exists: {output}")
        log_path = root / f"{name}.log"
        receipt.update(phase="running", current=name, updated_utc=timestamp())
        write_json(receipt_path, receipt)
        command = [sys.executable, str(repository / "scripts/run_practical_training.py"),
                   "--repository", str(repository), "--encoder", args.encoder, "--epsilon", "0.01",
                   "--retention-weight", "0.25", "--seed", str(seed), "--epochs", "12",
                   "--threads", str(args.threads), "--output", str(output)]
        print(json.dumps({"event": "replication_started", "encoder": args.encoder, "candidate": name}), flush=True)
        with log_path.open("x") as stream:
            result = subprocess.run(command, cwd=repository, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            receipt.update(phase="failed", returncode=result.returncode, updated_utc=timestamp())
            write_json(receipt_path, receipt)
            raise RuntimeError(f"Replication failed; inspect {log_path}")
        completion_path = output / "completion.json"
        completion = json.loads(completion_path.read_text())
        if completion["status"] != "completed_development_only" or completion["epochs"] != 12:
            raise ValueError("Replication did not complete all twelve epochs.")
        receipt["completed"].append({"candidate": name, "completion_sha256": sha(completion_path),
                                     "original_development_selection": completion["selection"],
                                     "final_epoch_selection": "pending_canonical_development_rescore", "completed_utc": timestamp()})
        receipt.update(updated_utc=timestamp())
        write_json(receipt_path, receipt)
        print(json.dumps({"event": "replication_complete", "encoder": args.encoder,
                          "candidate": name, "original_development_selection": completion["selection"]}), flush=True)
    receipt.update(phase="completed", current=None, updated_utc=timestamp())
    write_json(receipt_path, receipt)
    print(json.dumps({"event": "replication_queue_complete", "encoder": args.encoder}), flush=True)


if __name__ == "__main__":
    main()
