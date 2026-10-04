#!/usr/bin/env python3
"""Verify and summarize the complete train/development-only pair-scorer pilot.

This report preserves the original development inference results. Final model
selection must use the independent canonical inference rescore instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    root = repository / "results/practical_v6"
    rows, baselines = [], {}
    for encoder in ("vit_b32", "rn50"):
        for epsilon in (0.005, 0.01, 0.02):
            for retention in (0.25, 1.0):
                path = root / encoder / f"eps_{epsilon:g}_ret_{retention:g}_seed_17"
                completion = json.loads((path / "completion.json").read_text())
                ledger = json.loads((path / "ledger.json").read_text())
                history = json.loads((path / "history.json").read_text())
                selection = json.loads((path / "selection.json").read_text())
                expected = hashlib.sha256(json.dumps(ledger["identity"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                if expected != ledger["ledger_sha256"] or expected != completion["ledger_sha256"] or expected != history["ledger_sha256"]:
                    raise ValueError(f"Ledger identity mismatch: {path}")
                if completion["history_sha256"] != sha(path / "history.json") or completion["selection"] != selection:
                    raise ValueError(f"Completion does not bind final history/selection: {path}")
                epochs = history["epochs"]
                if [epoch["epoch"] for epoch in epochs] != list(range(13)) or completion["epochs"] != 12:
                    raise ValueError(f"Incomplete 12-epoch candidate: {path}")
                for epoch in epochs:
                    for record in (epoch["checkpoint"], epoch["retrieval"]["predictions"], epoch["composition"]["predictions"]):
                        if sha(path / record["path"]) != record["sha256"]:
                            raise ValueError(f"Changed checkpoint/development archive: {path / record['path']}")
                baseline = {"i2t_r1": epochs[0]["retrieval"]["i2t_r1"], "t2i_r1": epochs[0]["retrieval"]["t2i_r1"],
                            "paired_joint_accuracy": epochs[0]["composition"]["paired_joint_accuracy"]}
                if encoder in baselines and baseline != baselines[encoder]:
                    raise ValueError(f"Frozen development baselines disagree: {encoder}")
                baselines[encoder] = baseline
                row = {"encoder": encoder, "epsilon": epsilon, "retention_weight": retention,
                       "seed": 17, "completed_epochs": 12, "run": str(path.relative_to(repository)),
                       "ledger_sha256": expected, "completion_sha256": sha(path / "completion.json"),
                       "selection": selection}
                if selection["selected_epoch"] is not None:
                    chosen = epochs[selection["selected_epoch"]]
                    row["selected_update_norm"] = chosen["update_norm"]
                    row["selected_optimizer_steps"] = chosen["optimizer_steps"]
                    row["selected_paired_joint_accuracy"] = chosen["composition"]["paired_joint_accuracy"]
                    row["selected_mean_paired_joint_margin"] = chosen["composition"]["mean_paired_joint_margin"]
                rows.append(row)
    common = []
    for epsilon in (0.005, 0.01, 0.02):
        for retention in (0.25, 1.0):
            pair = [row for row in rows if row["epsilon"] == epsilon and row["retention_weight"] == retention]
            if all(row["selection"]["selected_epoch"] is not None for row in pair):
                gains = [row["selection"]["paired_joint_accuracy_gain"] for row in pair]
                common.append({"epsilon": epsilon, "retention_weight": retention,
                               "minimum_encoder_development_gain": min(gains), "mean_encoder_development_gain": sum(gains) / 2})
    common.sort(key=lambda row: (row["minimum_encoder_development_gain"], row["mean_encoder_development_gain"]), reverse=True)
    report = {"status": "complete_original_development_pilot_pending_canonical_rescore", "training_runs": 12,
              "epochs_per_run": 12, "checkpoint_count": 156, "raw_development_archives": 312,
              "held_out_benchmarks_read": False, "baselines": baselines, "candidates": rows,
              "common_configuration_descriptive_ranking": common,
              "selection_warning": "This ranking is descriptive and provisional. Canonical development inference must precede final selection. Development gain does not establish the practical goal."}
    output = args.output or root / "pilot_grid_summary_original.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("encoder epsilon retention epoch joint_gain_pp i2t_gain_pp t2i_gain_pp")
    for row in rows:
        selected = row["selection"]
        values = [row["encoder"], row["epsilon"], row["retention_weight"], selected["selected_epoch"]]
        values.extend(round(100 * selected[name], 5) if selected["selected_epoch"] is not None else None
                      for name in ("paired_joint_accuracy_gain", "i2t_r1_change", "t2i_r1_change"))
        print(" ".join(map(str, values)))
    print(json.dumps({"summary": str(output), "sha256": sha(output), "common_configuration_descriptive_ranking": common}, indent=2))


if __name__ == "__main__":
    main()
