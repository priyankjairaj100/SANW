#!/usr/bin/env python3
"""Independently verify AD raw predictions, all effects, and decision gates.

This script imports no project statistics or evaluation helpers. Bootstrap
resamples use a cluster multiplicity matrix, not the production gather kernel.
It verifies saved ranks; it does not reconstruct ranks from feature products.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ("vit_b32", "rn50")
FAMILIES = ("source", "supported", "allocation", "distilled", "allocation_distillation", "wise_ft")
DATASETS = ("e_vil_test1000", "visual_entailment", "sugarcrepe", "sugarcrepe_pp", "coco_karpathy")
ENDPOINTS = (("e_vil_test1000", "i2t.r1"), ("e_vil_test1000", "t2i.r1"),
             ("visual_entailment", "accuracy"), ("sugarcrepe_pp", "both_accuracy"))
SEEDS = (17, 29, 43)
REPLICATES = 100000
BOOTSTRAP_SEED = 20261004
EPSILON = 2e-12


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def flatten_numeric(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten_numeric(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = float(item)
    return result


def bootstrap_matrix(differences, cluster_ids, *, replicates=REPLICATES,
                     seed=BOOTSTRAP_SEED, batch_size=113):
    """Bootstrap several paired effects from cluster multiplicities.

    differences has shape [effect, seed, item]. Clusters have unequal item counts.
    Each resample recomputes its denominator. Seeds retain equal fixed weights.
    """
    values = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=str)
    if values.ndim != 3 or values.shape[2] != len(clusters) or not values.size or not np.isfinite(values).all():
        raise ValueError("Invalid paired effect tensor")
    if replicates < 1 or batch_size < 1:
        raise ValueError("Invalid bootstrap size")
    groups = defaultdict(list)
    for item, name in enumerate(clusters.tolist()):
        groups[name].append(item)
    ordered = [groups[name] for name in sorted(groups)]
    count = np.asarray([len(indices) for indices in ordered], dtype=np.float64)
    average = values.sum(axis=1) / values.shape[1]
    totals = np.stack([average[:, indices].sum(axis=1) for indices in ordered])
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, values.shape[0]), dtype=np.float64)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        selected = rng.integers(len(ordered), size=(size, len(ordered)))
        multiplicities = np.zeros((size, len(ordered)), dtype=np.float64)
        np.add.at(multiplicities, (np.arange(size)[:, None], selected), 1.)
        samples[start:start + size] = (multiplicities @ totals) / (multiplicities @ count)[:, None]
    return samples


def summarize_effect(delta, samples, cluster_ids, family_size):
    delta = np.asarray(delta, dtype=np.float64)
    by_seed = np.asarray([sum(row) / len(row) for row in delta])
    mean = float(delta.mean(axis=0).mean())
    tail = .05 / (2 * family_size)
    # Explicit linear interpolation independently implements NumPy's convention.
    ordered = np.sort(samples)
    def quantile(probability):
        position = probability * (len(ordered) - 1)
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        return float(ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
    return {"difference": mean, "ci_lower": quantile(tail), "ci_upper": quantile(1 - tail),
            "confidence": 1 - .05 / family_size, "family_size": family_size,
            "replicates": len(samples), "bootstrap_seed": BOOTSTRAP_SEED,
            "images": len(set(map(str, cluster_ids))), "items": delta.shape[1],
            "training_seeds": delta.shape[0], "seed_differences": by_seed.tolist(),
            "seed_difference_std": float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else None,
            "quantile_method": "linear"}


def reconstruct_metrics(dataset, raw):
    """Rebuild every numeric scoring aggregate and validate stored correctness."""
    if dataset in ("e_vil_test1000", "coco_karpathy"):
        metrics = {"images": len(raw["image_ids"]), "texts": len(raw["text_ids"])}
        for direction, limit in (("i2t", len(raw["text_ids"])), ("t2i", len(raw["image_ids"]))):
            ranks = raw[f"{direction}_ranks"]
            expected_length = len(raw["image_ids" if direction == "i2t" else "text_ids"])
            if (ranks.shape != (expected_length,) or not np.isfinite(ranks).all() or
                np.any(ranks != np.floor(ranks)) or np.any(ranks < 1) or np.any(ranks > limit)):
                raise ValueError("Saved retrieval ranks are invalid")
            ordered = sorted(map(int, ranks))
            count = len(ordered)
            metrics[direction] = {f"r{k}": sum(rank <= k for rank in ordered) / count for k in (1, 5, 10)}
            metrics[direction]["mean_rank"] = sum(ordered) / count
            metrics[direction]["median_rank"] = (ordered[(count - 1) // 2] + ordered[count // 2]) / 2
        metrics["mean_bidirectional_r1"] = (metrics["i2t"]["r1"] + metrics["t2i"]["r1"]) / 2
        return metrics
    if dataset == "visual_entailment":
        groups = defaultdict(lambda: defaultdict(list))
        identifiers = set()
        for iid, tid, relation, score in zip(raw["pair_image_ids"], raw["pair_text_ids"],
                                             raw["pair_relations"], raw["pair_scores"], strict=True):
            key = str(iid), str(tid), str(relation)
            if key in identifiers or relation not in ("supported", "contradicted") or not np.isfinite(score):
                raise ValueError("Invalid relation score record")
            identifiers.add(key)
            groups[str(iid)][str(relation)].append(float(score))
        accuracy, comparisons = [], []
        for iid in raw["image_ids"]:
            positives, negatives = groups[str(iid)]["supported"], groups[str(iid)]["contradicted"]
            if not positives or not negatives:
                raise ValueError("Missing relation comparison group")
            credit = [float(a > b) + .5 * float(a == b) for a in positives for b in negatives]
            accuracy.append(sum(credit) / len(credit))
            comparisons.append(len(credit))
        if not np.allclose(accuracy, raw["image_accuracy"], rtol=0, atol=EPSILON):
            raise ValueError("Relation correctness differs from saved scores")
        if not np.array_equal(comparisons, raw["comparison_counts"]):
            raise ValueError("Relation comparison counts differ")
        raw["image_accuracy"] = np.asarray(accuracy, dtype=np.float64)
        return {"accuracy": sum(accuracy) / len(accuracy), "eligible_images": len(accuracy),
                "manifest_images": len(accuracy), "comparison_count": sum(comparisons), "excluded_images": []}
    first = np.asarray([a > b for a, b in zip(raw["positive1_scores"], raw["negative_scores"], strict=True)])
    if not np.isfinite(raw["positive1_scores"]).all() or not np.isfinite(raw["negative_scores"]).all():
        raise ValueError("Nonfinite triplet scores")
    if not np.array_equal(first, raw["positive1_correct"]):
        raise ValueError("First-positive correctness differs from scores")
    metrics = {"items": len(first), "images": len(set(map(str, raw["image_ids"]))),
               "positive1_accuracy": sum(first) / len(first)}
    both = first
    if dataset == "sugarcrepe_pp":
        second = np.asarray([a > b for a, b in zip(raw["positive2_scores"], raw["negative_scores"], strict=True)])
        if not np.isfinite(raw["positive2_scores"]).all() or not np.array_equal(second, raw["positive2_correct"]):
            raise ValueError("Second-positive correctness differs from scores")
        both = first & second
        metrics.update(positive2_accuracy=sum(second) / len(second), both_accuracy=sum(both) / len(both))
    if not np.array_equal(both, raw["correct"]):
        raise ValueError("Triplet correctness differs from scores")
    metrics["accuracy"] = sum(both) / len(both)
    return metrics


def endpoint(raw, dataset, metric):
    if dataset == "e_vil_test1000":
        direction = metric.split(".")[0]
        return (np.asarray(raw[f"{direction}_ranks"] == 1, dtype=float),
                raw["image_ids" if direction == "i2t" else "text_ids"],
                raw["image_ids" if direction == "i2t" else "text_source_image_ids"])
    if dataset == "visual_entailment":
        return raw["image_accuracy"], raw["image_ids"], raw["image_ids"]
    if dataset == "sugarcrepe_pp":
        return np.asarray(raw["correct"], dtype=float), raw["item_ids"], raw["image_ids"]
    raise ValueError("Unknown primary endpoint")


def strategy_arrays(runs, predictions, dataset, metric):
    """Use equal draw means, followed by equal fixed-seed means."""
    anchor_ids = anchor_clusters = None
    per_seed = []
    seeds = (None,) if runs[0]["method"] == "frozen" else SEEDS
    for seed in seeds:
        selected = [run for run in runs if run["seed"] == seed]
        draw_ids = [run["draw_id"] for run in selected]
        if draw_ids != [None] and sorted(draw_ids) != [0, 1, 2]:
            raise ValueError("A strategy does not contain one state or three complete draws")
        values = []
        for run in sorted(selected, key=lambda value: -1 if value["draw_id"] is None else value["draw_id"]):
            vector, ids, clusters = endpoint(predictions[(run["state_id"], dataset)], dataset, metric)
            if anchor_ids is None:
                anchor_ids, anchor_clusters = ids, clusters
            if not np.array_equal(ids, anchor_ids) or not np.array_equal(clusters, anchor_clusters):
                raise ValueError("Paired prediction identities differ")
            values.append(vector)
        per_seed.append(sum(values) / len(values))
    if seeds == (None,):
        per_seed *= len(SEEDS)
    return np.stack(per_seed), anchor_ids, anchor_clusters


class Audit:
    def __init__(self, repository):
        self.repository = Path(repository).resolve()
        self.hash_cache = {}
        self.counts = Counter()
        self.maxima = defaultdict(float)

    def path(self, value):
        path = Path(value)
        path = (path if path.is_absolute() else self.repository / path).resolve()
        if not path.is_relative_to(self.repository):
            raise ValueError("Audit input escapes the repository")
        return path

    def check_hash(self, path, expected):
        path = self.path(path)
        if path not in self.hash_cache:
            self.hash_cache[path] = digest(path)
        if self.hash_cache[path] != expected:
            raise ValueError(f"Content hash mismatch: {path}")
        self.counts["hash_checks"] += 1

    def equal(self, actual, expected, category):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid numeric comparison: {category}")
        difference = float(np.max(np.abs(a - b))) if a.size else 0.
        self.maxima[category] = max(self.maxima[category], difference)
        self.counts[category] += a.size
        if difference > EPSILON:
            raise ValueError(f"Independent {category} mismatch: {difference}")

    def raw_identity(self, dataset, raw, manifest):
        images = [str(row["id"]) for row in manifest["images"] if row["split"] == "test"]
        if dataset in ("e_vil_test1000", "coco_karpathy", "visual_entailment"):
            if list(raw["image_ids"]) != images or len(set(images)) != len(images):
                raise ValueError("Prediction images differ from the test manifest")
        if dataset in ("e_vil_test1000", "coco_karpathy"):
            texts = [str(row["id"]) for row in manifest["texts"]]
            owner = {str(row["text_id"]): str(row["image_id"]) for row in manifest["pairs"] if row["relation"] == "source"}
            if list(raw["text_ids"]) != texts or list(raw["text_source_image_ids"]) != [owner[tid] for tid in texts]:
                raise ValueError("Retrieval identities or caption ownership differ")
        elif dataset == "visual_entailment":
            expected = {(str(row["image_id"]), str(row["text_id"]), row["relation"]) for row in manifest["pairs"]
                        if str(row["image_id"]) in set(images) and row["relation"] in ("supported", "contradicted")}
            actual = list(zip(map(str, raw["pair_image_ids"]), map(str, raw["pair_text_ids"]), map(str, raw["pair_relations"])))
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError("Relation pairs differ from the held-out manifest")
        else:
            for field, key in (("id", "item_ids"), ("image_id", "image_ids"), ("category", "categories"),
                               ("positive1_id", "positive1_ids"), ("negative_id", "negative_ids")):
                if list(raw[key]) != [str(row[field]) for row in manifest["triplets"]]:
                    raise ValueError("Triplet identities differ from the manifest")
            if dataset == "sugarcrepe_pp" and list(raw["positive2_ids"]) != [str(row["positive2_id"]) for row in manifest["triplets"]]:
                raise ValueError("Second-positive identities differ from the manifest")

    def load_index(self, index_path, dataset_config_path, lock, lock_hash, protocol_hash):
        index = read_json(self.path(index_path))
        if index["status"] != "complete" or index["protocol_sha256"] != protocol_hash:
            raise ValueError("Audit requires complete protocol-bound evaluation indices")
        encoder = index["encoder"]
        record = lock["encoders"][encoder]
        manifest_path = self.path(index["manifest"])
        self.check_hash(manifest_path, index["manifest_sha256"])
        if index["manifest_sha256"] != record["manifest_sha256"] or index["selection_lock_sha256"] != lock_hash:
            raise ValueError("Evaluation differs from the common selection lock")
        manifest = read_json(manifest_path)
        if not manifest["matched_controls_complete"] or manifest["test_outcomes_used_for_selection"]:
            raise ValueError("Selection or matched-control completion is invalid")
        self.check_hash(index["prescore_receipt"], index["prescore_receipt_sha256"])
        receipt = read_json(self.path(index["prescore_receipt"]))
        for key in ("protocol_sha256", "encoder", "manifest_sha256", "selection_lock_sha256",
                    "source_hashes", "selections", "decomposition_selections", "evidence_type"):
            if index[key] != receipt[key]:
                raise ValueError(f"Index differs from prescore receipt: {key}")
        for name, expected in receipt["source_hashes"].items():
            self.check_hash(name, expected)
        ledger_path = manifest_path.parent / "ledger.json"
        self.check_hash(ledger_path, receipt["ledger_file_sha256"])
        ledger = read_json(ledger_path)
        if canonical_digest(ledger["identity"]) != ledger["ledger_sha256"] or ledger["ledger_sha256"] != manifest["ledger_sha256"]:
            raise ValueError("Training ledger identity differs")
        for name, expected in ledger["identity"]["source_sha256"].items():
            self.check_hash(name, expected)
        for section in ("inputs", "development_retrieval", "assignments"):
            for value in ledger["identity"][section].values():
                if isinstance(value, dict) and "path" in value and "sha256" in value:
                    self.check_hash(value["path"], value["sha256"])
        for label in ("primary", "sensitivity"):
            self.check_hash(manifest_path.parent / f"selection_{label}.json", manifest["selection_sha256"][label])
        if manifest["selection_sha256"] != record["selection_sha256"]:
            raise ValueError("Selections differ from the shared lock")
        self.check_hash(dataset_config_path, receipt["dataset_config_sha256"])
        config = read_json(self.path(dataset_config_path))
        if config["encoder"] != encoder or set(config["datasets"]) != set(DATASETS):
            raise ValueError("Invalid encoder dataset config")
        manifests = {}
        for dataset, paths in config["datasets"].items():
            for kind, path in paths.items():
                self.check_hash(path, receipt["input_hashes"][dataset][f"{kind}_sha256"])
            manifests[dataset] = read_json(self.path(paths["manifest"]))
        actual_states = [{key: value for key, value in run.items() if key != "datasets"} for run in index["runs"]]
        if actual_states != receipt["runs"]:
            raise ValueError("Scored states differ from the prescore plan")
        all_states = {state["state_id"]: state for state in manifest["states"]}
        if len(all_states) != len(manifest["states"]):
            raise ValueError("The training manifest contains duplicate states")
        ids = {row["state_id"] for row in manifest["selections"] + manifest["decomposition_selections"]}
        if ({run["state_id"] for run in index["runs"]} != ids | {"frozen"} or
            len(index["runs"]) != len(ids) + 1):
            raise ValueError("Scoring must include exactly the selected and decomposition states")
        if index["selections"] != manifest["selections"] or index["decomposition_selections"] != manifest["decomposition_selections"]:
            raise ValueError("Scoring selection plan differs from the manifest")
        predictions, metrics = {}, {}
        import torch
        for run in index["runs"]:
            state = {key: value for key, value in run.items() if key != "datasets"}
            if run["state_id"] != "frozen":
                if state != all_states[run["state_id"]]:
                    raise ValueError("Scored state differs from the development manifest")
                self.check_hash(run["checkpoint"], run["checkpoint_sha256"])
                checkpoint = torch.load(self.path(run["checkpoint"]), map_location="cpu", weights_only=True)
                for key in ("method", "seed", "epoch", "learning_rate"):
                    if checkpoint[key] != state[key]:
                        raise ValueError("Checkpoint header differs from its state")
                if checkpoint["protocol_sha256"] != protocol_hash or checkpoint["ledger_sha256"] != manifest["ledger_sha256"]:
                    raise ValueError("Checkpoint provenance differs")
                norm = np.sqrt(sum(np.square(weight.numpy().astype(np.float64)).sum()
                                   for weight in checkpoint["state_dict"].values()))
                self.equal(norm, state["update_norm"], "checkpoint_norms")
            if set(run["datasets"]) != set(DATASETS):
                raise ValueError("A scored state lacks a benchmark")
            for dataset, artifact in run["datasets"].items():
                self.check_hash(artifact["predictions"], artifact["predictions_sha256"])
                self.check_hash(artifact["metadata"], artifact["metadata_sha256"])
                metadata = read_json(self.path(artifact["metadata"]))
                if (metadata["provenance"] != {"run": state, "dataset": dataset,
                    "prescore_receipt_sha256": index["prescore_receipt_sha256"]} or
                    metadata["predictions_sha256"] != artifact["predictions_sha256"] or
                    metadata["metrics"] != artifact["metrics"]):
                    raise ValueError("Prediction provenance differs from the plan")
                with np.load(self.path(artifact["predictions"]), allow_pickle=False) as archive:
                    raw = {name: archive[name] for name in archive.files}
                self.raw_identity(dataset, raw, manifests[dataset])
                rebuilt = flatten_numeric(reconstruct_metrics(dataset, raw))
                published = flatten_numeric(artifact["metrics"])
                if set(rebuilt) != set(published):
                    raise ValueError("Raw aggregate field sets differ")
                for key, value in rebuilt.items():
                    self.equal(value, published[key], "raw_aggregate_endpoints")
                predictions[(run["state_id"], dataset)] = raw
                metrics[(run["state_id"], dataset)] = rebuilt
                self.counts["prediction_archives"] += 1
            self.counts["states"] += 1
        return index, predictions, metrics


def selected(index, family, tolerance=1.):
    states = {run["state_id"]: run for run in index["runs"]}
    if family == "frozen":
        return [states["frozen"]]
    entries = [row for row in index["selections"] if row["family"] == family and row["tolerance_pp"] == tolerance]
    expected = Counter((seed, draw) for seed in SEEDS for draw in range(3)) if family == "matched_allocation_distillation" else Counter((seed, None) for seed in SEEDS)
    runs = [states[row["state_id"]] for row in entries]
    if Counter((run["seed"], run["draw_id"]) for run in runs) != expected:
        raise ValueError("Selected strategy lacks the exact seed and draw grid")
    for entry, state in zip(entries, runs, strict=True):
        if entry["seed"] != state["seed"] or entry.get("draw_id") != state["draw_id"]:
            raise ValueError("Selection identity differs from state identity")
    return runs


def build_effects(index, predictions):
    encoder = index["encoder"]
    states = {run["state_id"]: run for run in index["runs"]}
    joint = {row["seed"]: row for row in selected(index, "allocation_distillation")}
    for control in selected(index, "matched_allocation_distillation"):
        reference = joint[control["seed"]]
        if any(control[key] != reference[key] for key in ("epoch", "learning_rate", "source_mix", "beta")):
            raise ValueError("Random controls do not match the joint schedule")
    cells = defaultdict(list)
    for entry in index["decomposition_selections"]:
        state = states[entry["state_id"]]
        reference = joint[entry["seed"]]
        cell = entry["cell"]
        if cell not in ("supported", "allocation", "distilled", "allocation_distillation"):
            raise ValueError("Unknown decomposition cell")
        if any(state[key] != reference[key] for key in ("epoch", "learning_rate", "seed")):
            raise ValueError("Decomposition schedule differs from selected AD")
        mix = reference["source_mix"] if cell in ("allocation", "allocation_distillation") else 0.
        beta = reference["beta"] if cell in ("distilled", "allocation_distillation") else 0.
        if state["source_mix"] != mix or state["beta"] != beta or state["family"] != cell or state["draw_id"] is not None:
            raise ValueError("Decomposition parameters differ")
        cells[cell].append(state)
    if any(Counter(row["seed"] for row in cells[cell]) != Counter(SEEDS)
           for cell in ("supported", "allocation", "distilled", "allocation_distillation")):
        raise ValueError("Incomplete decomposition cells")
    result = []
    pairs = [(family, "frozen") for family in FAMILIES] + [("allocation_distillation", family)
        for family in ("distilled", "allocation", "wise_ft", "matched_allocation_distillation")]
    for dataset, metric in ENDPOINTS:
        strategies = {family: strategy_arrays(selected(index, family), predictions, dataset, metric)
                      for family in FAMILIES + ("frozen", "matched_allocation_distillation")}
        def align(a, b):
            if not np.array_equal(a[1], b[1]) or not np.array_equal(a[2], b[2]):
                raise ValueError("Contrast strategies have different item identities")
        for left, right in pairs:
            a, b = strategies[left], strategies[right]
            align(a, b)
            result.append({"effect_id": f"{encoder}__{left}_minus_{right}__{dataset}__{metric}",
                "encoder": encoder, "left": left, "right": right, "dataset": dataset, "metric": metric,
                "family_size": 80, "delta": a[0] - b[0], "clusters": a[2]})
        arrays = {cell: strategy_arrays(runs, predictions, dataset, metric) for cell, runs in cells.items()}
        for value in arrays.values():
            align(value, strategies["frozen"])
        u, a, d, ad = (arrays[cell][0] for cell in ("supported", "allocation", "distilled", "allocation_distillation"))
        effects = {"allocation_main": (a + ad - u - d) / 2,
                   "distillation_main": (d + ad - u - a) / 2,
                   "interaction": ad + u - a - d}
        for name, delta in effects.items():
            result.append({"effect_id": f"{encoder}__{name}__{dataset}__{metric}",
                "encoder": encoder, "effect": name, "dataset": dataset, "metric": metric,
                "family_size": 24, "delta": delta, "clusters": arrays["supported"][2]})
    return result


def gate_checks(runs, contrasts):
    outcomes = {(row["dataset"], row["metric"]): row for row in contrasts}
    return {"all_three_nonzero": len(runs) == 3 and {run["seed"] for run in runs} == set(SEEDS)
            and all(run["epoch"] > 0 and run["update_norm"] > 0 for run in runs),
            "i2t_retention": outcomes[("e_vil_test1000", "i2t.r1")]["ci_lower"] > -.01,
            "t2i_retention": outcomes[("e_vil_test1000", "t2i.r1")]["ci_lower"] > -.01,
            "sugarcrepe_pp_improvement": outcomes[("sugarcrepe_pp", "both_accuracy")]["ci_lower"] > 0.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs=2, required=True)
    parser.add_argument("--dataset-configs", nargs=2, required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = Audit(ROOT)
    audit.check_hash(args.protocol, args.protocol_sha256)
    protocol = read_json(audit.path(args.protocol))
    specification = protocol["evaluation"]
    if (specification["inference"]["bootstrap_replicates"] != REPLICATES or
        specification["inference"]["bootstrap_seed"] != BOOTSTRAP_SEED or
        specification["primary_contrasts"]["family_size"] != 80 or specification["decomposition"]["family_size"] != 24 or
        specification["primary_endpoints"] != [{"dataset": d, "metric": m} for d, m in ENDPOINTS]):
        raise ValueError("Frozen inferential design differs from the independent audit")
    lock = read_json(audit.path(args.selection_lock))
    lock_hash = digest(audit.path(args.selection_lock))
    if lock["protocol_sha256"] != args.protocol_sha256 or set(lock["encoders"]) != set(ENCODERS):
        raise ValueError("Invalid common selection lock")
    configs = {read_json(audit.path(path))["encoder"]: path for path in args.dataset_configs}
    data, effects = {}, []
    for path in args.indices:
        encoder = read_json(audit.path(path))["encoder"]
        if encoder in data:
            raise ValueError("Duplicate encoder index")
        data[encoder] = audit.load_index(path, configs[encoder], lock, lock_hash, args.protocol_sha256)
        index, predictions, _ = data[encoder]
        effects.extend(build_effects(index, predictions))
        print(json.dumps({"raw_audit_complete": encoder, "states": len(index["runs"])}), flush=True)
    if set(data) != set(ENCODERS) or Counter(row["family_size"] for row in effects) != {80: 80, 24: 24}:
        raise ValueError("The independent effect grid is incomplete")
    analysis = audit.path(args.analysis)
    published = {}
    for name, count in (("primary_contrasts.json", 80), ("decomposition_contrasts.json", 24)):
        document = read_json(analysis / name)
        if document["protocol_sha256"] != args.protocol_sha256 or document["family_size"] != count or len(document["contrasts"]) != count:
            raise ValueError("Published contrast family is incomplete")
        for row in document["contrasts"]:
            if row["effect_id"] in published:
                raise ValueError("Duplicate published effect")
            published[row["effect_id"]] = row
    if set(published) != {row["effect_id"] for row in effects}:
        raise ValueError("Published effects differ from the independently reconstructed grid")
    receipt = read_json(analysis / "preanalysis_receipt.json")
    if receipt["protocol_sha256"] != args.protocol_sha256:
        raise ValueError("Analysis receipt protocol differs")
    for name, expected in {**receipt["source_hashes"], **receipt["indices"]}.items():
        audit.check_hash(name, expected)
    production_audit = read_json(analysis / "analysis_audit.json")
    audit.check_hash(analysis / "preanalysis_receipt.json", production_audit["preanalysis_receipt_sha256"])
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row["dataset"], row["metric"], tuple(map(str, row["clusters"])))].append(row)
    reconstructed = {}
    with np.load(analysis / "bootstrap_samples.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(published):
            raise ValueError("Saved bootstrap arrays have missing or additional effects")
        for (dataset, metric, _), group in grouped.items():
            draws = bootstrap_matrix([row["delta"] for row in group], group[0]["clusters"])
            for column, row in enumerate(group):
                key = row["effect_id"]
                samples = draws[:, column]
                audit.equal(samples, saved[key], "bootstrap_samples")
                summary = summarize_effect(row["delta"], samples, row["clusters"], row["family_size"])
                for field, value in summary.items():
                    if isinstance(value, (int, float, list)):
                        audit.equal(value, published[key][field], "effect_statistics")
                    elif value != published[key][field]:
                        raise ValueError("A published effect metadata field differs")
                for field in ("encoder", "dataset", "metric", "left", "right", "effect"):
                    if field in row and published[key][field] != row[field]:
                        raise ValueError("Published effect identity differs")
                reconstructed[key] = {**row, **summary}
            print(json.dumps({"bootstrap_audit_complete": dataset, "metric": metric,
                              "effects": len(group), "replicates": REPLICATES}), flush=True)
    gates_document = read_json(analysis / "practical_success_gates.json")
    gate_records = {(row["encoder"], row["family"]): row for row in gates_document["gates"]}
    if set(gate_records) != {(encoder, family) for encoder in ENCODERS for family in FAMILIES}:
        raise ValueError("Practical gate grid differs")
    for (encoder, family), reported in gate_records.items():
        rows = [row for row in reconstructed.values() if row["encoder"] == encoder and
                row.get("left") == family and row.get("right") == "frozen"]
        checks = gate_checks(selected(data[encoder][0], family), rows)
        if checks != reported["checks"] or all(checks.values()) != reported["passed"]:
            raise ValueError("A practical gate differs from independently reconstructed intervals")
        audit.counts["practical_gates"] += 1
    for row in gates_document["cross_encoder"]:
        if row["passed_both_encoders"] != all(gate_records[(encoder, row["family"])]["passed"] for encoder in ENCODERS):
            raise ValueError("Cross-encoder gate differs")
        audit.counts["cross_encoder_gates"] += 1
    if Counter(row["family"] for row in gates_document["cross_encoder"]) != Counter(FAMILIES):
        raise ValueError("Cross-encoder gate families are incomplete")
    # Check all descriptive reports directly against per-state raw aggregates.
    descriptions = read_json(analysis / "selected_strategy_metrics.json")["strategies"]
    seen = set()
    for description in descriptions:
        encoder, family, tolerance = (description[key] for key in ("encoder", "family", "tolerance_pp"))
        key = encoder, family, tolerance
        if key in seen:
            raise ValueError("Duplicate strategy description")
        seen.add(key)
        index, _, metrics = data[encoder]
        runs = selected(index, family, tolerance)
        if {r["state_id"] for r in description["states"]} != {r["state_id"] for r in runs}:
            raise ValueError("Strategy description references different states")
        for label, numbers in description["metrics"].items():
            dataset, metric = label.split(".", 1)
            seeds = (None,) if family == "frozen" else SEEDS
            means = [float(np.mean([metrics[(run["state_id"], dataset)][metric]
                     for run in runs if run["seed"] == seed])) for seed in seeds]
            audit.equal(means, numbers["seed_means"], "descriptive_statistics")
            audit.equal(np.mean(means), numbers["mean"], "descriptive_statistics")
            if len(means) > 1:
                audit.equal(np.std(means, ddof=1), numbers["seed_std"], "descriptive_statistics")
            elif numbers["seed_std"] is not None:
                raise ValueError("Frozen standard deviation must be null")
    expected_descriptions = {(encoder, family, tolerance) for encoder in ENCODERS for family in FAMILIES for tolerance in (0., 1.)}
    expected_descriptions |= {(encoder, "frozen", None) for encoder in ENCODERS}
    expected_descriptions |= {(encoder, "matched_allocation_distillation", 1.) for encoder in ENCODERS}
    if seen != expected_descriptions:
        raise ValueError("Descriptive strategies are incomplete")
    csv_seen = set()
    with (analysis / "all_state_metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["encoder"], row["state_id"], row["dataset"], row["metric"]
            if key in csv_seen:
                raise ValueError("Duplicate state metric row")
            csv_seen.add(key)
            audit.equal(float(row["value"]), data[row["encoder"]][2][(row["state_id"], row["dataset"])][row["metric"]], "csv_endpoints")
    expected_csv = set()
    for encoder, (index, _, metrics) in data.items():
        for (sid, dataset), values in metrics.items():
            for metric in values:
                if metric.endswith("accuracy") or metric.startswith(("i2t.", "t2i.")) or metric == "mean_bidirectional_r1":
                    expected_csv.add((encoder, sid, dataset, metric))
    if csv_seen != expected_csv:
        raise ValueError("CSV report lacks expected raw endpoint rows")
    output = audit.path(args.output)
    evidence_paths = args.indices + args.dataset_configs + [args.protocol, args.selection_lock]
    evidence_paths += [str(path.relative_to(ROOT)) for path in analysis.iterdir()
                       if path.is_file() and path.resolve() != output]
    result = {"schema_version": 1, "status": "passed", "protocol_sha256": args.protocol_sha256,
        "scope": "saved ranks, score-derived correctness, receipts, all 104 effects, all 100000 draws per effect, gates, and descriptive reports",
        "excluded_scope": "independent feature extraction and recomputation of retrieval ranks from feature dot products",
        "independent_implementation": "no project evaluator or statistics imports; cluster multiplicity matrix and explicit quantile interpolation",
        "primary_effects": 80, "decomposition_effects": 24, "replicates_per_effect": REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED, "counts": dict(audit.counts), "maximum_absolute_differences": dict(audit.maxima),
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__)},
        "input_sha256": {str(audit.path(path).relative_to(ROOT)): digest(audit.path(path)) for path in evidence_paths},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    if output.exists() and read_json(output) != result:
        raise ValueError("Existing independent audit receipt differs; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
