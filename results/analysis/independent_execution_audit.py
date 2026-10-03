"""Independent verification of completed evaluation, without importing its metrics.

Executed only after all selection and evaluation outputs are frozen. This script
does not fit, select or change a model and does not import historical results.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_predictions(run, dataset):
    path = ROOT / run["datasets"][dataset]["predictions"]
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    index_path = ROOT / "results/evaluation/index.json"
    index = json.loads(index_path.read_text())
    assert index["status"] == "complete" and index["complete_selected_study"]
    by_method = {}
    for run in index["runs"]:
        by_method.setdefault(run["method"], []).append(run)
    for runs in by_method.values():
        runs.sort(key=lambda r: r["seed"] or 0)
    contrast_data = json.loads((ROOT / "results/analysis/primary_contrasts.json").read_text())
    bootstrap_audits = []
    with np.load(ROOT / "results/analysis/primary_bootstrap_samples.npz", allow_pickle=False) as saved:
        for contrast in contrast_data["contrasts"]:
            dataset = contrast["dataset"]
            pa = [load_predictions(run, dataset) for run in by_method[contrast["method_a"]]]
            pb = [load_predictions(run, dataset) for run in by_method[contrast["method_b"]]]
            key = "image_accuracy" if dataset == "visual_entailment" else "correct"
            ids = pa[0]["image_ids"]
            point_values = np.array([[float(a) - float(b) for a, b in zip(x[key], y[key])]
                                     for x, y in zip(pa, pb)]).mean(axis=0)
            groups = {}
            for j, iid in enumerate(ids):
                groups.setdefault(str(iid), []).append(j)
            ordered = sorted(groups)
            cluster_sums = np.array([sum(float(point_values[j]) for j in groups[iid]) for iid in ordered])
            cluster_sizes = np.array([len(groups[iid]) for iid in ordered])
            rng = np.random.default_rng(contrast["bootstrap_seed"])
            independent = []
            for _ in range(contrast["replicates"] // 100):
                draws = rng.integers(0, len(ordered), size=(100, len(ordered)))
                # Independent multiplicity-weighted computation rather than the
                # evaluator's direct indexing of sampled sum/count vectors.
                multiplicity = np.stack([np.bincount(row, minlength=len(ordered)) for row in draws])
                independent.extend((multiplicity @ cluster_sums) / (multiplicity @ cluster_sizes))
            independent = np.asarray(independent)
            assert len(independent) == contrast["replicates"]
            sample_error = float(np.max(np.abs(independent - saved[contrast["effect_id"]])))
            point = sum(float(v) for v in point_values) / len(point_values)
            quantiles = np.percentile(independent, [100 * .05 / 12, 100 * (1 - .05 / 12)], method="linear")
            point_error = abs(point - contrast["difference"])
            interval_error = float(np.max(np.abs(quantiles - [contrast["ci_lower"], contrast["ci_upper"]])))
            assert max(sample_error, point_error, interval_error) < 1e-12
            bootstrap_audits.append({"effect_id": contrast["effect_id"], "point_error": point_error,
                                     "replicate_max_error": sample_error, "interval_max_error": interval_error})
    # Verify exact identity of every epoch-zero prediction field, not merely
    # their rounded summary numbers.
    zero_checks = []
    frozen = by_method["frozen"][0]
    for run in index["runs"]:
        if run["method"] == "frozen" or run["epoch"] != 0:
            continue
        for dataset in frozen["datasets"]:
            expected, actual = load_predictions(frozen, dataset), load_predictions(run, dataset)
            assert expected.keys() == actual.keys()
            for key in expected:
                assert np.array_equal(expected[key], actual[key]), (run["run_id"], dataset, key)
        zero_checks.append(run["run_id"])
    # Re-score a deterministic sample of 64 queries per direction in three
    # independently chosen states, sorting every candidate directly.
    manifest = json.loads((ROOT / "data/coco_karpathy/manifest.json").read_text())
    with np.load(ROOT / "results/features/coco_karpathy/features.npz", allow_pickle=False) as z:
        original_images, original_texts = z["image_features"], z["text_features"]
        image_ids, text_ids = list(z["image_ids"]), list(z["text_ids"])
    image_lookup, text_lookup = {v: j for j, v in enumerate(image_ids)}, {v: j for j, v in enumerate(text_ids)}
    image_relevance, text_relevance = [set() for _ in image_ids], [set() for _ in text_ids]
    for pair in manifest["pairs"]:
        ii, ti = image_lookup[pair["image_id"]], text_lookup[pair["text_id"]]
        image_relevance[ii].add(ti)
        text_relevance[ti].add(ii)
    retrieval_checks = []
    for run in [frozen, by_method["grounded"][0], by_method["multipositive"][0]]:
        state = None if run["method"] == "frozen" else torch.load(ROOT / run["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
        adapted = []
        for original, direction in [(original_images, "image"), (original_texts, "text")]:
            pieces = []
            with torch.inference_mode():
                for start in range(0, len(original), 1024):
                    x = torch.from_numpy(np.ascontiguousarray(original[start:start + 1024], dtype=np.float32))
                    if state is not None:
                        x = x + F.linear(x, state[f"{direction}.weight"])
                    pieces.append(F.normalize(x, dim=-1).numpy().astype(np.float64))
            adapted.append(np.concatenate(pieces))
        images, texts = adapted
        p = load_predictions(run, "coco_karpathy")
        for direction, query_features, candidates, relevance in [
            ("i2t", images, texts, image_relevance), ("t2i", texts, images, text_relevance)
        ]:
            queries = np.random.default_rng(20261003).choice(len(query_features), 64, replace=False)
            rows = query_features[queries] @ candidates.T
            for qi, row in zip(queries, rows):
                ordered = np.argsort(-row, kind="stable")
                rank = next(j + 1 for j, ci in enumerate(ordered) if ci in relevance[qi])
                assert rank == int(p[f"{direction}_ranks"][qi]), (run["run_id"], direction, int(qi))
                assert np.array_equal(ordered[:10], p[f"{direction}_top_indices"][qi])
            retrieval_checks.append({"run_id": run["run_id"], "direction": direction,
                                     "full_pool_queries_checked": len(queries), "candidates_per_query": len(candidates)})
    receipt = {"status": "passed", "source_index_sha256": digest(index_path),
               "audit_source_sha256": digest(Path(__file__)), "bootstrap_effects": bootstrap_audits,
               "epoch_zero_states_bitwise_equal_to_frozen": zero_checks,
               "full_pool_retrieval_rescoring": retrieval_checks,
               "historical_aggregates_read": False}
    (ROOT / "results/analysis/independent_execution_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
