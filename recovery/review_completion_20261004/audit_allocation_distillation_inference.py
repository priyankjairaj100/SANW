#!/usr/bin/env python3
"""Independent synthetic checks for paired inference. No test data are read."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gcr.allocation_distillation_analysis import (
    decomposition_arrays, effect, equal_seed_draw_values, practical_gate,
)


def main():
    item_ids = np.array(["a0", "b0", "b1", "c0", "c1", "c2"])
    cluster_ids = np.array(["a", "b", "b", "c", "c", "c"])
    generator = np.random.default_rng(72317)
    source = generator.integers(0, 2, size=(3, 3, 6)).astype(float)
    predictions, runs = {}, []
    for seed_index, seed in enumerate((17, 29, 43)):
        for draw in range(3):
            state_id = f"synthetic_{seed}_{draw}"
            runs.append({"state_id": state_id, "seed": seed, "draw_id": draw})
            predictions[(state_id, "sugarcrepe_pp")] = {
                "item_ids": item_ids, "image_ids": cluster_ids,
                "correct": source[seed_index, draw],
            }
    actual = equal_seed_draw_values(runs, predictions, "sugarcrepe_pp", "both_accuracy",
                                    expected_draws=(0, 1, 2))
    expected_seed_values = np.empty((3, 6))
    for i in range(3):
        for j in range(6):
            expected_seed_values[i, j] = sum(float(source[i, d, j]) for d in range(3)) / 3
    np.testing.assert_array_equal(actual[0], expected_seed_values)
    for bad_runs in (runs[:-1], runs + [runs[0]]):
        try:
            equal_seed_draw_values(bad_runs, predictions, "sugarcrepe_pp", "both_accuracy",
                                   expected_draws=(0, 1, 2))
        except ValueError:
            pass
        else:
            raise AssertionError("Missing or duplicate assignment draws were accepted")

    baseline = (generator.integers(0, 2, size=(3, 6)).astype(float), item_ids, cluster_ids)
    max_error = 0.0
    for family_size in (80, 24):
        result, samples = effect(actual, baseline, family_size=family_size,
                                 replicates=1009, seed=717)
        rng = np.random.default_rng(717)
        groups = [[0], [1, 2], [3, 4, 5]]
        expected = []
        for _ in range(1009):
            selected_groups = rng.integers(0, 3, size=3)
            items = [j for group in selected_groups for j in groups[group]]
            observations = [float(actual[0][seed, item] - baseline[0][seed, item])
                            for item in items for seed in range(3)]
            expected.append(sum(observations) / len(observations))
        np.testing.assert_allclose(samples, expected, rtol=0, atol=3e-16)
        quantiles = np.quantile(expected, [.05/(2*family_size), 1-.05/(2*family_size)], method="linear")
        np.testing.assert_allclose([result["ci_lower"], result["ci_upper"]], quantiles,
                                   rtol=0, atol=3e-16)
        max_error = max(max_error, float(np.max(np.abs(samples - expected))))

    u = (np.zeros((3, 6)), item_ids, cluster_ids)
    a = (np.full((3, 6), .2), item_ids, cluster_ids)
    d = (np.full((3, 6), .3), item_ids, cluster_ids)
    ad = (np.full((3, 6), .8), item_ids, cluster_ids)
    decomposed = decomposition_arrays(u, a, d, ad)
    for name, expected in (("allocation_main", .35), ("distillation_main", .45), ("interaction", .3)):
        np.testing.assert_allclose(decomposed[name][0], expected, rtol=0, atol=2e-16)

    valid_runs = [{"seed": s, "epoch": 1, "update_norm": 1e-3} for s in (17, 29, 43)]
    contrast_rows = [
        {"dataset": "e_vil_test1000", "metric": "i2t.r1", "family_size": 80, "ci_lower": -.009},
        {"dataset": "e_vil_test1000", "metric": "t2i.r1", "family_size": 80, "ci_lower": -.009},
        {"dataset": "sugarcrepe_pp", "metric": "both_accuracy", "family_size": 80, "ci_lower": .001},
    ]
    assert practical_gate(valid_runs, contrast_rows)["passed"]
    for index, threshold in enumerate((-.01, -.01, 0.0)):
        altered = [dict(row) for row in contrast_rows]
        altered[index]["ci_lower"] = threshold
        assert not practical_gate(valid_runs, altered)["passed"]
    for key in ("epoch", "update_norm"):
        altered_runs = [dict(run) for run in valid_runs]
        altered_runs[0][key] = 0
        assert not practical_gate(altered_runs, contrast_rows)["passed"]

    paths = ["src/gcr/allocation_distillation_analysis.py", "src/gcr/evaluation.py"]
    report = {"passed": True, "scope": "synthetic aggregation, resampling, factorial effects, strict gates",
              "test_data_read": False, "scientific_training_performed": False,
              "bootstrap_checks": 2018, "maximum_bootstrap_error": max_error,
              "draw_averaging": "three equally weighted draws within each of three seeds",
              "cluster_sizes": [1, 2, 3], "correction_families_checked": [80, 24],
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    output = Path(__file__).with_name("allocation_distillation_independent_inference_checks.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
