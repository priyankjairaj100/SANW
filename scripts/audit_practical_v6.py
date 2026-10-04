#!/usr/bin/env python3
"""Independent full-gallery, score, paired-cluster, and gate audit for practical v6.

Imports no project scorer, metric, or statistical implementation. Reconstructs the
residual from checkpoint tensors. Recomputes exact top-one over bounded candidate
sets and every SC++ pair, then all 100,000 bootstrap values for all ten effects.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ("vit_b32", "rn50")
SEEDS = (17, 29, 43)
DATASETS = ("e_vil_test1000", "coco_karpathy", "sugarcrepe_pp")
NBOOT, RNG_SEED, FAMILY = 100000, 20261007, 80


def path(v):
    p = Path(v)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Path escapes repository")
    return p


def digest(p):
    h = hashlib.sha256()
    with path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(path(p).read_text())


def check_hash(p, expected):
    if digest(p) != expected:
        raise ValueError(f"Hash mismatch: {p}")


class Audit:
    def __init__(self):
        self.counts = defaultdict(int)
        self.max_score_error = 0.
        self.max_bootstrap_error = 0.

    def equal(self, actual, expected, name, tolerance=2e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {name}")
        if a.dtype.kind in "USb" or b.dtype.kind in "USb" or a.dtype.kind in "iu" and b.dtype.kind in "iu":
            if not np.array_equal(a, b):
                raise ValueError(f"Exact mismatch: {name}")
        elif not np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=False):
            raise ValueError(f"Numeric mismatch: {name}")
        self.counts[name] += a.size

    def residual(self, images, texts, checkpoint):
        if checkpoint is None:
            return np.zeros(len(images), dtype=np.float64)
        weights = checkpoint["state_dict"]
        config = checkpoint.get("model_config", checkpoint.get("config"))
        dimension, rank = config["dimension"], config["rank"]
        values = np.empty(len(images), dtype=np.float64)
        with torch.inference_mode():
            for start in range(len(images)):
                image = torch.tensor(images[start:start + 1], dtype=torch.float32)
                text = torch.tensor(texts[start:start + 1], dtype=torch.float32)
                x = image - weights["image_mean"]
                y = text - weights["text_mean"]
                first = math.sqrt(dimension) * x * y
                xi = F.linear(x, weights["image_projection.weight"]) * math.sqrt(dimension / rank)
                yt = F.linear(y, weights["text_projection.weight"]) * math.sqrt(dimension / rank)
                second = math.sqrt(rank) * xi * yt
                cosine = torch.sum(image * text, dim=1, keepdim=True)
                features = torch.cat([first, second, cosine], dim=1)
                hidden = F.gelu(F.linear(features, weights["pair_hidden.weight"], weights["pair_hidden.bias"]))
                value = F.linear(hidden, weights["output.weight"], weights["output.bias"]).reshape(-1)
                values[start] = float(torch.tanh(value[0]).double() * float(config["epsilon"]))
        if not np.isfinite(values).all() or (np.abs(values) > float(config["epsilon"])).any():
            raise ValueError("Independent bounded score failed")
        self.counts["recomputed_residual_pairs"] += len(values)
        return values

    def load_data(self, entry, expected_hashes, encoder):
        for key, filename in entry.items():
            check_hash(filename, expected_hashes[key])
        manifest = read(entry["manifest"])
        with np.load(path(entry["features"]), allow_pickle=False) as archive:
            values = {k: archive[k] for k in ("image_features", "text_features", "image_ids", "text_ids")}
        for kind in ("image", "text"):
            self.equal(values[f"{kind}_ids"], np.asarray([str(r["id"]) for r in manifest[f"{kind}s"]]), "feature_manifest_identity")
            matrix = values[f"{kind}_features"]
            if matrix.shape != (len(values[f"{kind}_ids"]), 512 if encoder == "vit_b32" else 1024) or not np.isfinite(matrix).all():
                raise ValueError("Invalid feature archive")
            # Reproduce the frozen preprocessing, independently of model code.
            tensor = torch.as_tensor(matrix, dtype=torch.float32)
            norms = torch.linalg.vector_norm(tensor, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
            values[f"{kind}_features"] = (tensor / norms).numpy()
        chosen = [i for i, row in enumerate(manifest["images"]) if row["split"] == "test"]
        values["image_features"], values["image_ids"] = values["image_features"][chosen], values["image_ids"][chosen]
        return manifest, values

    def retrieval(self, raw, manifest, arrays, checkpoint, metrics):
        images, texts = arrays["image_features"], arrays["text_features"]
        self.equal(raw["image_ids"], arrays["image_ids"], "prediction_identity")
        self.equal(raw["text_ids"], arrays["text_ids"], "prediction_identity")
        ilook = {str(i): k for k, i in enumerate(arrays["image_ids"])}
        tlook = {str(t): k for k, t in enumerate(arrays["text_ids"])}
        owners = np.full(len(texts), -1, dtype=np.int64)
        image_counts = np.zeros(len(images), dtype=np.int64)
        for pair in manifest["pairs"]:
            ti, ii = tlook[str(pair["text_id"])], ilook[str(pair["image_id"])]
            if pair["relation"] != "source" or owners[ti] != -1:
                raise ValueError("Duplicate or invalid source ownership")
            owners[ti] = ii
            image_counts[ii] += 1
        if (owners < 0).any() or not (image_counts == 5).all():
            raise ValueError("Source ownership incomplete")
        self.equal(raw["text_source_image_ids"], arrays["image_ids"][owners], "prediction_identity")
        eps = 0. if checkpoint is None else float(checkpoint.get("model_config", checkpoint.get("config"))["epsilon"])
        for direction, query, gallery in (("i2t", images, texts), ("t2i", texts, images)):
            winner = np.empty(len(query), dtype=np.int64)
            winner_score = np.empty(len(query), dtype=np.float64)
            for start in range(0, len(query), 37):
                # A wider envelope and different block shape audit shortlist safety.
                frozen = np.dot(query[start:start + 37].astype(np.float64), gallery.astype(np.float64).T)
                row, col = np.where(frozen >= frozen.max(axis=1, keepdims=True) - 2 * eps - 2e-8)
                qi = row + start
                correction = np.empty(len(qi), dtype=np.float64)
                for offset in range(0, len(qi), 8192):
                    qidx, gidx = qi[offset:offset + 8192], col[offset:offset + 8192]
                    left, right = (query[qidx], gallery[gidx]) if direction == "i2t" else (gallery[gidx], query[qidx])
                    correction[offset:offset + len(qidx)] = self.residual(left, right, checkpoint)
                corrected = frozen[row, col] + correction
                for local in range(len(frozen)):
                    positions = np.flatnonzero(row == local)
                    # Explicit Python tuple ordering independently enforces score/index ties.
                    selected = min(positions, key=lambda p: (-corrected[p], col[p]))
                    winner[start + local] = col[selected]
                    winner_score[start + local] = corrected[selected]
            self.equal(winner, raw[f"{direction}_top_indices"], "full_gallery_top1")
            error = float(np.max(np.abs(winner_score - raw[f"{direction}_top_scores"])))
            self.max_score_error = max(self.max_score_error, error)
            self.equal(winner_score, raw[f"{direction}_top_scores"], "retrieval_scores", tolerance=2e-8)
            correct = owners[winner] == np.arange(len(images)) if direction == "i2t" else winner == owners
            self.equal(correct, raw[f"{direction}_correct"], "retrieval_correctness")
            self.equal(sum(correct) / len(correct), metrics[direction]["r1"], "metric_aggregates")
            self.equal(np.sum(raw[f"{direction}_candidate_counts"]), metrics[direction]["scored_pairs"], "metric_aggregates")
            self.equal(np.max(raw[f"{direction}_candidate_counts"]), metrics[direction]["max_candidates"], "metric_aggregates")

    def triplets(self, raw, manifest, arrays, checkpoint, metrics):
        triplets = manifest["triplets"]
        ilook = {str(i): k for k, i in enumerate(arrays["image_ids"])}
        tlook = {str(t): k for k, t in enumerate(arrays["text_ids"])}
        for key, source in (("item_ids", "id"), ("image_ids", "image_id"), ("categories", "category")):
            self.equal(raw[key], np.asarray([str(r[source]) for r in triplets]), "prediction_identity")
        image = arrays["image_features"][[ilook[str(row["image_id"])] for row in triplets]]
        scores = {}
        for field in ("positive1", "positive2", "negative"):
            ids = np.asarray([str(r[f"{field}_id"]) for r in triplets])
            self.equal(raw[f"{field}_ids"], ids, "prediction_identity")
            text = arrays["text_features"][[tlook[t] for t in ids]]
            score = np.sum(image.astype(np.float64) * text.astype(np.float64), axis=1) + self.residual(image, text, checkpoint)
            self.equal(score, raw[f"{field}_scores"], "triplet_scores", tolerance=2e-8)
            self.max_score_error = max(self.max_score_error, float(np.max(np.abs(score - raw[f"{field}_scores"]))))
            scores[field] = score
        first, second = scores["positive1"] > scores["negative"], scores["positive2"] > scores["negative"]
        for field, value in (("positive1_correct", first), ("positive2_correct", second), ("correct", first & second)):
            self.equal(value, raw[field], "triplet_correctness")
        for name, correct in (("positive1_accuracy", first), ("positive2_accuracy", second), ("both_accuracy", first & second)):
            self.equal(sum(correct) / len(correct), metrics[name], "metric_aggregates")

    def bootstrap(self, differences, clusters):
        grouped = defaultdict(list)
        for item, cluster in enumerate(clusters.tolist()):
            grouped[str(cluster)].append(item)
        ordered = [grouped[k] for k in sorted(grouped)]
        seed_average = sum(differences) / len(differences)
        totals = np.asarray([sum(seed_average[items]) for items in ordered])
        counts = np.asarray([len(items) for items in ordered])
        rng = np.random.default_rng(RNG_SEED)
        draws = np.empty(NBOOT)
        for start in range(0, NBOOT, 113):
            size = min(113, NBOOT - start)
            selected = rng.integers(len(ordered), size=(size, len(ordered)))
            multiplicity = np.zeros((size, len(ordered)), dtype=np.float64)
            np.add.at(multiplicity, (np.arange(size)[:, None], selected), 1.)
            draws[start:start + size] = (multiplicity @ totals) / (multiplicity @ counts)
        return draws

    def run(self, analysis_path):
        analysis = read(analysis_path)
        if set(analysis["indices"]) != set(ENCODERS) or len(analysis["effects"]) != 10:
            raise ValueError("Complete cross-encoder study required")
        check_hash(analysis["bootstrap_samples"], analysis["bootstrap_samples_sha256"])
        with np.load(path(analysis["bootstrap_samples"]), allow_pickle=False) as z:
            saved_samples = {key: z[key] for key in z.files}
        rebuilt_effects = {}
        for encoder, index_record in analysis["indices"].items():
            check_hash(index_record["path"], index_record["sha256"])
            index = read(index_record["path"])
            if index["status"] != "complete" or index["encoder"] != encoder or len(index["runs"]) != 4:
                raise ValueError("Incomplete scoring index")
            check_hash(index["prescore_receipt"], index["prescore_receipt_sha256"])
            receipt = read(index["prescore_receipt"])
            check_hash(receipt["selection_lock"], receipt["selection_lock_sha256"])
            lock = read(receipt["selection_lock"])
            if receipt["selection_lock_sha256"] != analysis["selection_lock_sha256"] or lock["family"] != analysis["family"]:
                raise ValueError("Cross-encoder lock differs")
            if lock["family_size"] != FAMILY or lock["bootstrap_replicates"] != NBOOT or lock["bootstrap_seed"] != RNG_SEED:
                raise ValueError("Inference contract changed")
            check_hash(lock["protocol"], lock["protocol_sha256"])
            check_hash(receipt["manifest"], receipt["manifest_sha256"])
            if receipt["manifest_sha256"] != lock["encoders"][encoder]["manifest_sha256"]:
                raise ValueError("Selected manifest changed")
            manifest = read(receipt["manifest"])
            if manifest["protocol_sha256"] != lock["protocol_sha256"] or manifest["test_outcomes_used_for_selection"] or manifest["current_test_outcomes_used_for_selection"]:
                raise ValueError("Selection provenance invalid")
            if sorted(s["seed"] for s in manifest["states"]) != list(SEEDS) or len(manifest["states"]) != 3:
                raise ValueError("Selected seed grid incomplete")
            if any(s["epoch"] <= 0 or s["update_norm"] <= 0 for s in manifest["states"]):
                raise ValueError("Nonzero trained states required")
            for filename, expected in {**lock["source_hashes"], **manifest["source_hashes"]}.items():
                check_hash(filename, expected)
            check_hash(manifest["training_metadata"]["path"], manifest["training_metadata"]["sha256"])
            training_meta = read(manifest["training_metadata"]["path"])
            check_hash(receipt["dataset_config"], receipt["dataset_config_sha256"])
            config = read(receipt["dataset_config"])
            if config["encoder"] != encoder:
                raise ValueError("Dataset encoder differs")
            data = {name: self.load_data(config["datasets"][name], receipt["input_hashes"][name], encoder) for name in DATASETS}
            for name in DATASETS:
                metadata = read(config["datasets"][name]["metadata"])
                for key in ("model_revision", "weights_sha256", "open_clip_version", "logit_scale"):
                    if metadata.get(key) != training_meta[key]:
                        raise ValueError("Frozen feature provenance differs")
            if [r["state"] for r in index["runs"]] != receipt["states"] or receipt["states"][1:] != manifest["states"]:
                raise ValueError("Scored states differ from locked selection")
            raw_by_state = {}
            for run in index["runs"]:
                state = run["state"]
                checkpoint = None
                if state["checkpoint"]:
                    check_hash(state["checkpoint"], state["checkpoint_sha256"])
                    checkpoint = torch.load(path(state["checkpoint"]), map_location="cpu", weights_only=True)
                    for key in ("seed", "epoch"):
                        if checkpoint[key] != state[key]:
                            raise ValueError("Checkpoint header mismatch")
                    if checkpoint.get("optimizer_steps", 0) <= 0 or abs(checkpoint.get("update_norm", -1) - state["update_norm"]) > 1e-12:
                        raise ValueError("Checkpoint update evidence differs")
                    output = checkpoint["state_dict"]["output.weight"]
                    if not torch.isfinite(output).all() or not torch.any(output != 0):
                        raise ValueError("No fitted output update")
                if set(run["datasets"]) != set(DATASETS):
                    raise ValueError("Incomplete benchmark plan")
                for name, artifact in run["datasets"].items():
                    check_hash(artifact["predictions"], artifact["predictions_sha256"])
                    check_hash(artifact["metadata"], artifact["metadata_sha256"])
                    metadata = read(artifact["metadata"])
                    if metadata["metrics"] != artifact["metrics"] or metadata["predictions_sha256"] != artifact["predictions_sha256"]:
                        raise ValueError("Raw archive metadata mismatch")
                    if metadata["provenance"] != {"state": state, "dataset": name, "prescore_receipt_sha256": index["prescore_receipt_sha256"]}:
                        raise ValueError("Prediction provenance mismatch")
                    with np.load(path(artifact["predictions"]), allow_pickle=False) as z:
                        raw = {k: z[k] for k in z.files}
                    ds_manifest, arrays = data[name]
                    if name == "sugarcrepe_pp":
                        self.triplets(raw, ds_manifest, arrays, checkpoint, artifact["metrics"])
                    else:
                        self.retrieval(raw, ds_manifest, arrays, checkpoint, artifact["metrics"])
                    raw_by_state[(state["state_id"], name)] = raw
                    self.counts["prediction_archives"] += 1
                self.counts["states"] += 1
            selected = sorted(manifest["states"], key=lambda s: s["seed"])
            for name in DATASETS:
                frozen = raw_by_state[("frozen", name)]
                selected_raw = [raw_by_state[(s["state_id"], name)] for s in selected]
                for metric in (("both_accuracy",) if name == "sugarcrepe_pp" else ("i2t.r1", "t2i.r1")):
                    field = "correct" if metric == "both_accuracy" else metric.split(".")[0] + "_correct"
                    cluster_key = "text_source_image_ids" if metric == "t2i.r1" else "image_ids"
                    delta = np.array([raw[field].astype(float) - frozen[field].astype(float) for raw in selected_raw])
                    eid = f"{encoder}__{name}__{metric.replace('.', '_')}"
                    actual = self.bootstrap(delta, frozen[cluster_key])
                    self.equal(actual, saved_samples[eid], "bootstrap_values", tolerance=2e-12)
                    self.max_bootstrap_error = max(self.max_bootstrap_error, float(np.max(np.abs(actual - saved_samples[eid]))))
                    order = sorted(actual.tolist())
                    def quantile(prob):
                        at = prob * (NBOOT - 1)
                        lo, hi = int(math.floor(at)), int(math.ceil(at))
                        return order[lo] + (at - lo) * (order[hi] - order[lo])
                    low, high = quantile(.05 / (2 * FAMILY)), quantile(1 - .05 / (2 * FAMILY))
                    effect = next(r for r in analysis["effects"] if r["effect_id"] == eid)
                    for key, value in {"difference": delta.mean(), "ci_lower": low, "ci_upper": high,
                                       "seed_differences": delta.mean(axis=1), "frozen": frozen[field].mean(),
                                       "selected": np.array([r[field] for r in selected_raw]).mean()}.items():
                        self.equal(value, effect[key], "effect_aggregates", tolerance=2e-12)
                    if effect["family_size"] != FAMILY or effect["replicates"] != NBOOT or effect["bootstrap_seed"] != RNG_SEED:
                        raise ValueError("Effect inference contract mismatch")
                    rebuilt_effects[(encoder, name, metric)] = low
                    self.counts["effects"] += 1
        gates = []
        for encoder in ENCODERS:
            retain = all(rebuilt_effects[(encoder, "e_vil_test1000", m)] > -.01 for m in ("i2t.r1", "t2i.r1"))
            improve = rebuilt_effects[(encoder, "sugarcrepe_pp", "both_accuracy")] > 0
            coco = all(rebuilt_effects[(encoder, "coco_karpathy", m)] > -.01 for m in ("i2t.r1", "t2i.r1"))
            gates.append({"encoder": encoder, "nonzero_three_seeds": True, "retrieval_retention": retain,
                          "caption_improvement": improve, "passed": retain and improve, "secondary_coco_retention": coco})
        if gates != analysis["gates"] or analysis["cross_encoder_passed"] != all(g["passed"] for g in gates):
            raise ValueError("Practical gate verdict mismatch")
        if analysis["cross_encoder_with_coco_passed"] != all(g["passed"] and g["secondary_coco_retention"] for g in gates):
            raise ValueError("COCO transfer verdict mismatch")
        return {"schema_version": 1, "passed": True, "analysis": str(path(analysis_path).relative_to(ROOT)),
                "analysis_sha256": digest(analysis_path), "source_sha256": digest(__file__), "counts": dict(self.counts),
                "max_score_difference": self.max_score_error, "max_bootstrap_difference": self.max_bootstrap_error,
                "independence": "no project scorer, metric, or statistics helpers imported; full-gallery top-one and every pair recomputed",
                "gates": gates, "evaluation_status": "exploratory_after_historical_test_reuse",
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--torch-threads", type=int, choices=[1], default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    result = Audit().run(args.analysis)
    destination = path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
