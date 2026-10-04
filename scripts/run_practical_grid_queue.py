#!/usr/bin/env python3
"""Run the five remaining declared development pilot candidates serially.

The queue waits for the already-running epsilon=.01, retention=1 candidate.
It binds the scorer, trainer and protocol before launching further candidates.
No benchmark evaluation is performed by this helper.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("vit_b32", "rn50"), required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.threads <= 3:
        raise ValueError("This queue permits at most three CPU threads.")
    repository = args.repository.resolve()
    root = repository / "results/practical_v6" / args.encoder
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "pilot_queue.json"
    if receipt_path.exists():
        raise FileExistsError("A queue receipt already exists; inspect it before another launch.")
    bound_names = ["src/gcr/practical_scorer.py", "src/gcr/practical_training.py", "scripts/run_practical_training.py",
                   "results/practical_v6/protocol_v1.json"]
    sources = {name: digest(repository / name) for name in bound_names}
    candidates = [(epsilon, weight) for epsilon in (0.005, 0.01, 0.02) for weight in (0.25, 1.0)
                  if (epsilon, weight) != (0.01, 1.0)]
    receipt = {"schema": "sanw_practical_pilot_queue_v1", "encoder": args.encoder,
               "created_utc": timestamp(), "phase": "waiting_for_initial_candidate", "source_sha256": sources,
               "threads": args.threads, "completed": [], "planned": candidates,
               "test_evaluation": "none"}
    atomic_json(receipt_path, receipt)
    dependency = root / "eps_0.01_ret_1_seed_17" / "completion.json"
    print(json.dumps({"event": "waiting_for_initial_candidate", "encoder": args.encoder, "dependency": str(dependency)}), flush=True)
    deadline = time.monotonic() + 4 * 3600
    while not dependency.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Initial candidate did not complete within four hours.")
        time.sleep(3)
    initial = json.loads(dependency.read_text())
    if initial.get("status") != "completed_development_only" or initial.get("epochs") != 12:
        raise ValueError("Initial candidate completion is invalid.")
    receipt["initial_completion_sha256"] = digest(dependency)
    for epsilon, weight in candidates:
        for name, expected in sources.items():
            if digest(repository / name) != expected:
                raise ValueError(f"Bound source changed before a fit: {name}")
        name = f"eps_{epsilon:g}_ret_{weight:g}_seed_17"
        output = root / name
        if output.exists():
            raise FileExistsError(f"Candidate already exists: {output}")
        log_path = root / f"{name}.log"
        receipt.update(phase="running", current=name, updated_utc=timestamp())
        atomic_json(receipt_path, receipt)
        command = [sys.executable, str(repository / "scripts/run_practical_training.py"),
                   "--repository", str(repository), "--encoder", args.encoder,
                   "--epsilon", str(epsilon), "--retention-weight", str(weight), "--seed", "17",
                   "--epochs", "12", "--threads", str(args.threads), "--output", str(output)]
        print(json.dumps({"event": "starting_candidate", "encoder": args.encoder, "candidate": name}), flush=True)
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            receipt.update(phase="failed", returncode=result.returncode, log=str(log_path.relative_to(repository)), updated_utc=timestamp())
            atomic_json(receipt_path, receipt)
            raise RuntimeError(f"Candidate failed; see {log_path}")
        completion_path = output / "completion.json"
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "completed_development_only" or completion.get("epochs") != 12:
            raise ValueError("Candidate ended without complete development execution.")
        receipt["completed"].append({"candidate": name, "completion_sha256": digest(completion_path),
                                     "selection": completion["selection"], "completed_utc": timestamp()})
        receipt.update(updated_utc=timestamp())
        atomic_json(receipt_path, receipt)
        print(json.dumps({"event": "candidate_complete", "encoder": args.encoder, "candidate": name,
                          "selection": completion["selection"]}), flush=True)
    receipt.update(phase="completed", current=None, updated_utc=timestamp())
    atomic_json(receipt_path, receipt)
    print(json.dumps({"event": "queue_complete", "encoder": args.encoder, "new_candidates": len(candidates)}), flush=True)


if __name__ == "__main__":
    main()
