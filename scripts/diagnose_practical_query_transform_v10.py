#!/usr/bin/env python3
"""Training-only deployment arithmetic/timing diagnosis of a completed v10 state.

This is NOT an evaluation scorer, new checkpoint, or accuracy experiment. It
uses fixed evenly spaced training queries and the complete source galleries.
Run alone after fitting/auditing; benchmark numbers include no encoder cost.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gcr.practical_constrained_v8 import ConstrainedBilinearScorer
from gcr.practical_constrained_evaluation_v8 import CanonicalScorer, canonical_pair_scores
from gcr.practical_training_data_v10 import digest, verify_record


class QueryTransform:
    """No gallery transformation, query normalization, or D-by-D B materialization."""

    def __init__(self, model):
        self.mu_v, self.mu_t = model.image_mean.copy(), model.text_mean.copy()
        self.ua = model.image_basis @ model.coefficient
        self.v = model.text_basis.copy()
        self.text_offset_factor = self.v.T @ self.mu_t
        self.image_offset_factor = self.ua.T @ self.mu_v

    def image_queries(self, images):
        coordinates = (images - self.mu_v) @ self.ua
        return images + coordinates @ self.v.T, -(coordinates @ self.text_offset_factor)

    def text_queries(self, texts):
        coordinates = (texts - self.mu_t) @ self.v
        return texts + coordinates @ self.ua.T, -(coordinates @ self.image_offset_factor)


def fixed_query_indices(size, count):
    if size < 1 or count < 1:
        raise ValueError("Query size/count must be positive")
    count = min(size, count)
    # Integer formula is platform-independent and includes both endpoints.
    return np.array([0] if count == 1 else [i * (size - 1) // (count - 1) for i in range(count)], dtype=np.int64)


def source_layout(manifest, expected_images=6000):
    images, texts = manifest["images"], manifest["texts"]
    image_ids, text_ids = [x["id"] for x in images], [x["id"] for x in texts]
    if (len(images) != expected_images or len(set(image_ids)) != len(images)
            or len(set(text_ids)) != len(texts) or any(x["split"] != "train" for x in images)):
        raise ValueError("Require distinct, exclusively training owners and text IDs")
    ilook, tlook = {x: i for i, x in enumerate(image_ids)}, {x: i for i, x in enumerate(text_ids)}
    associated, sources = set(), {}
    for pair in manifest["pairs"]:
        if (pair["image_id"] not in ilook or pair["text_id"] not in tlook
                or pair["relation"] not in ("source", "supported", "contradicted", "neutral")):
            raise ValueError("Invalid training association")
        j = tlook[pair["text_id"]]
        if j in associated:
            raise ValueError("Caption ID has multiple associations")
        associated.add(j)
        if pair["relation"] == "source":
            sources[j] = ilook[pair["image_id"]]
    rows = np.array(sorted(sources), dtype=np.int64)
    owners = np.array([sources[j] for j in rows], dtype=np.int64)
    if len(associated) != len(texts) or not np.all(np.bincount(owners, minlength=len(images)) == 5):
        raise ValueError("Require complete ownership and exactly five source captions per owner")
    return image_ids, text_ids, rows


def verify_completed_state(repository, protocol_path, protocol_sha, completion_path):
    """Check fixed-budget completion and selected epoch; never select a new state."""
    if digest(protocol_path) != protocol_sha:
        raise ValueError("Protocol hash differs")
    protocol = json.loads(protocol_path.read_text())
    completion = json.loads(completion_path.read_text())
    ledger_path = completion_path.parent / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    identity = ledger["identity"]
    ledger_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    encoder, family = identity["encoder"], identity["family"]
    config = identity["config"]
    if (protocol.get("study") != "sanw_practical_v10" or encoder not in protocol["encoders"]
            or family not in ("joint", "no_retention") or config["seed"] not in protocol["seeds"]
            or config != {**protocol["fit_config"], "seed": config["seed"]}
            or ledger.get("ledger_sha256") != ledger_sha or completion.get("ledger_sha256") != ledger_sha
            or any(x.get("study") != "sanw_practical_v10" or x.get("mode") != "full"
                   or x.get("encoder") != encoder or x.get("family") != family
                   or x.get("protocol_sha256") != protocol_sha or x.get("config") != config for x in (identity, completion))):
        raise ValueError("Completed fit identity differs")
    provenance = identity["training_provenance"]
    if (identity["source_sha256"] != protocol["source_sha256"]
            or provenance["inputs"] != protocol["training_inputs"][encoder]
            or provenance.get("heldout_used") is not False or provenance.get("fit_split") != "train"
            or identity.get("official_development_or_benchmarks_used") is not False
            or identity.get("fit_gallery_image_count") != 6000 or identity.get("fit_gallery_text_count") != 30000):
        raise ValueError("Training provenance differs")
    watched = {str(protocol_path.resolve()): protocol_sha, str(completion_path.resolve()): digest(completion_path),
               str(ledger_path.resolve()): digest(ledger_path), str(Path(__file__).resolve()): digest(__file__)}
    for name, sha in protocol["source_sha256"].items():
        watched[str((repository / name).resolve())] = sha
    verify_unchanged(watched)
    history = completion["history"]
    steps_per_epoch = (6000 + config["batch_size"] - 1) // config["batch_size"]
    if ([row["epoch"] for row in history] != list(range(1, config["epochs"] + 1))
            or completion["optimizer_steps"] != config["epochs"] * steps_per_epoch
            or any(row["optimizer_steps"] != row["epoch"] * steps_per_epoch
                   or not np.isfinite(row["training_objective"]) for row in history)
            or [{k: v for k, v in row.items() if k != "checkpoint"} for row in completion["checkpoint_history"]] != history):
        raise ValueError("Fixed training budget/history was not completed")
    eligible = [row for row in history if row["nonzero"]]
    if not eligible:
        raise ValueError("No nonzero selected state")
    best = min(eligible, key=lambda row: (row["training_objective"], row["epoch"]))
    if (completion["selected_epoch"] != best["epoch"] or completion["selected_training_objective"] != best["training_objective"]):
        raise ValueError("Saved state is not its prescribed training-objective selection")
    if family == "joint" and any(not all(row["certificate"].get(key) is True for key in
            ("ranking_checked_canonically", "ranking_preserved", "feasible_with_tolerance")) for row in history):
        raise ValueError("Joint history lacks its recorded finite-training certificate")
    selected_record = completion["selected_checkpoint"]
    epoch_record = completion["checkpoint_history"][best["epoch"] - 1]["checkpoint"]
    if epoch_record["ledger_sha256"] != ledger_sha:
        raise ValueError("Epoch ledger identity differs")
    selected = (completion_path.parent / selected_record["path"]).resolve()
    epoch = (completion_path.parent / epoch_record["path"]).resolve()
    for path, item in ((selected, selected_record), (epoch, epoch_record)):
        watched[str(path)] = item["sha256"]
    verify_unchanged(watched)
    with np.load(selected, allow_pickle=False) as a, np.load(epoch, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Selected state differs from its recorded epoch")
    model = ConstrainedBilinearScorer.load(selected)
    if (model.coefficient.shape != (config["rank"], config["rank"])
            or not np.any(model.coefficient != 0) or not np.isfinite(model.coefficient).all()
            or np.linalg.norm(model.coefficient) > config["radius"] * (1 + 128 * np.finfo(np.float64).eps)):
        raise ValueError("Invalid selected coefficient")
    return protocol, completion, identity, model, watched


def verify_unchanged(records):
    for name, sha in records.items():
        if digest(name) != sha:
            raise ValueError(f"Source/input/state hash differs: {name}")


def load_training_galleries(repository, protocol, identity, watched):
    """Read only the separate train6000 archive, never the old mixed-split bank."""
    import torch
    import torch.nn.functional as F

    records = protocol["training_inputs"][identity["encoder"]]
    if set(records) != {"manifest", "features", "metadata"}:
        raise ValueError("Require the three locked training input records")
    paths = {key: verify_record(repository, value) for key, value in records.items()}
    watched.update({str(paths[k]): records[k]["sha256"] for k in paths})
    manifest, metadata = (json.loads(paths[k].read_text()) for k in ("manifest", "metadata"))
    image_ids, text_ids, source_rows = source_layout(manifest)
    provenance = identity["training_provenance"]
    dimension = 512 if identity["encoder"] == "vit_b32" else 1024
    if (metadata.get("all_rows_training_only") is not True or metadata.get("fresh_confirmation_rows") != 0
            or metadata.get("manifest_sha256") != records["manifest"]["sha256"]
            or metadata.get("features_sha256") != records["features"]["sha256"]
            or metadata.get("dimension") != dimension or metadata.get("dtype") != "float32"
            or metadata.get("normalization") != "L2" or metadata.get("image_count") != len(image_ids)
            or metadata.get("text_count") != len(text_ids)
            or provenance["training_image_manifest_indices"] != list(range(len(image_ids)))
            or provenance["training_text_manifest_indices"] != list(range(len(text_ids)))
            or provenance["training_source_text_manifest_indices"] != source_rows.tolist()
            or provenance["torch_version"] != torch.__version__):
        raise ValueError("Train-only metadata, normalization runtime, or gallery order differs")
    with np.load(paths["features"], allow_pickle=False) as archive:
        if (archive["image_ids"].tolist() != image_ids or archive["text_ids"].tolist() != text_ids):
            raise ValueError("Training archive order differs")
        images32, all_texts32 = archive["image_features"], archive["text_features"]
    if (images32.dtype != np.float32 or all_texts32.dtype != np.float32
            or images32.shape != (len(image_ids), dimension) or all_texts32.shape != (len(text_ids), dimension)):
        raise ValueError("Training archive dtype/shape differs")
    texts32 = all_texts32[source_rows]
    del all_texts32
    for values in (images32, texts32):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-4, rtol=2e-4):
            raise ValueError("Training gallery has invalid raw normalized features")
    images = F.normalize(torch.from_numpy(images32), dim=-1).numpy().astype(np.float64)
    texts = F.normalize(torch.from_numpy(texts32), dim=-1).numpy().astype(np.float64)
    # Owned immutable arrays catch accidental in-place changes in diagnosis.
    images.flags.writeable = texts.flags.writeable = False
    return images, texts, image_ids, [text_ids[j] for j in source_rows], source_rows


def compare_direction(direction, images, texts, model, transform, queries, gallery_block=512, near_tie_absolute=1e-10):
    """Canonical pair reductions on EVERY gallery entry for each sampled query."""
    if direction not in ("i2t", "t2i") or gallery_block < 1:
        raise ValueError("Unknown direction or invalid gallery block")
    canonical = CanonicalScorer(model.image_mean, model.text_mean, model.image_basis, model.text_basis, model.coefficient)
    _, _, left, right = canonical.prepare(images, texts)
    gallery = texts if direction == "i2t" else images
    q, offsets = (transform.image_queries(images[queries]) if direction == "i2t" else transform.text_queries(texts[queries]))
    # Match the exact transformed batch GEMM shape used in the timing case.
    # At the CLI limit this is <=128x30000 doubles; canonical products remain
    # gallery-blocked, so no all-training-query square score matrix is formed.
    raw_matrix = q @ gallery.T
    rows = []
    total_squared_error, max_error, total_pairs = 0., 0., 0
    for pos, query in enumerate(queries):
        raw, exact = raw_matrix[pos], np.empty(len(gallery))
        for start in range(0, len(gallery), gallery_block):
            stop = min(start + gallery_block, len(gallery))
            sl = slice(start, stop)
            n = stop - start
            if direction == "i2t":
                exact[sl] = canonical_pair_scores(np.broadcast_to(images[query], (n, images.shape[1])), texts[sl],
                                                  np.broadcast_to(left[query], (n, left.shape[1])), right[sl])
            else:
                exact[sl] = canonical_pair_scores(images[sl], np.broadcast_to(texts[query], (n, texts.shape[1])),
                                                  left[sl], np.broadcast_to(right[query], (n, right.shape[1])))
        restored = raw + offsets[pos]
        error = restored - exact
        current_max = float(np.max(np.abs(error)))
        max_error = max(max_error, current_max)
        total_squared_error += float(error @ error)
        total_pairs += len(gallery)
        top = int(exact.argmax())
        top_two = np.partition(exact, -2)[-2:] if len(exact) > 1 else None
        margin = float(top_two.max() - top_two.min()) if top_two is not None else None
        raw_top, restored_top = int(raw.argmax()), int(restored.argmax())
        rows.append({"query_gallery_index": int(query), "canonical_top_index": top, "transformed_raw_top_index": raw_top,
                     "offset_restored_top_index": restored_top, "canonical_top_two_margin": margin,
                     "max_absolute_score_error_after_offset": current_max,
                     "raw_top_disagrees": raw_top != top, "offset_restored_top_disagrees": restored_top != top,
                     "canonical_score_regret_of_raw_top": float(exact[top] - exact[raw_top]),
                     "canonical_score_regret_of_offset_restored_top": float(exact[top] - exact[restored_top]),
                     "near_tie": margin is not None and margin <= max(near_tie_absolute, 2 * current_max),
                     "query_offset": float(offsets[pos])})
    return {"direction": direction, "query_count": len(rows), "gallery_count": len(gallery), "pairs_compared": total_pairs,
            "max_absolute_score_error_after_offset": max_error, "rms_score_error_after_offset": (total_squared_error / total_pairs)**.5,
            "raw_top_disagreements": sum(r["raw_top_disagrees"] for r in rows),
            "offset_restored_top_disagreements": sum(r["offset_restored_top_disagrees"] for r in rows),
            "near_tie_queries": sum(r["near_tie"] for r in rows), "queries": rows,
            "transformed_scoring": "one query-subset-by-complete-gallery GEMM, matching the benchmark batch shape",
            "tie_policy": "first ascending gallery manifest index among numerically equal scores",
            "near_tie_rule": f"canonical top-two margin <= max({near_tie_absolute}, 2 * observed max absolute error for that query)",
            "scope": "descriptive arithmetic comparison; observed errors are not a universal numerical bound; no replacements/rescoring"}


def timed(call, warmups, repeats):
    for _ in range(warmups):
        call()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return {"seconds": durations, "median_seconds": float(np.median(durations)),
            "minimum_seconds": min(durations), "maximum_seconds": max(durations)}


def benchmark_direction(direction, images, texts, transform, queries, warmups, repeats):
    gallery = texts if direction == "i2t" else images
    values = images[queries] if direction == "i2t" else texts[queries]
    operation = transform.image_queries if direction == "i2t" else transform.text_queries
    transformed, offsets = operation(values)
    def end_to_end():
        q, c = operation(values)
        return q @ gallery.T + c[:, None]
    cases = {"frozen_full_gallery_gemm": lambda: values @ gallery.T,
             "query_transform_and_offset_only": lambda: operation(values),
             "transformed_full_gallery_gemm_only": lambda: transformed @ gallery.T,
             "transformed_full_scores_end_to_end": end_to_end}
    result = {key: timed(call, warmups, repeats) for key, call in cases.items()}
    for value in result.values():
        value["amortized_seconds_per_query"] = value["median_seconds"] / len(queries)
    return {"direction": direction, "query_batch_size": len(queries), "gallery_count": len(gallery),
            "timings": result, "end_to_end_over_frozen_median_ratio": result["transformed_full_scores_end_to_end"]["median_seconds"] / result["frozen_full_gallery_gemm"]["median_seconds"],
            "scope": "warm CPU float64 exact dense scoring, fixed case order; no encoder/IO/ANN/top-k/canonical-rescoring cost; amortized batch cost is not single-query latency"}


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def diagnose(args):
    from threadpoolctl import threadpool_info, threadpool_limits
    import torch

    if any(os.environ.get(k) != str(args.threads) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("Launch with all three BLAS/OMP environment thread counts equal to --threads")
    if args.queries < 1 or args.queries > 128 or args.warmups < 1 or args.repeats < 3 or args.threads < 1 or args.gallery_block < 1:
        raise ValueError("Use 1..128 queries, >=1 warmup, >=3 repeats, and positive threads/block")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    protocol, completion, identity, model, watched = verify_completed_state(args.repository, args.protocol, args.protocol_sha256, args.completion)
    images, texts, image_ids, text_ids, source_rows = load_training_galleries(args.repository, protocol, identity, watched)
    model_before = {k: array_hash(getattr(model, k)) for k in ("image_mean", "text_mean", "image_basis", "text_basis", "coefficient")}
    gallery_before = {"images": array_hash(images), "texts": array_hash(texts)}
    loading_seconds = time.perf_counter() - started
    iquery, tquery = fixed_query_indices(len(images), args.queries), fixed_query_indices(len(texts), args.queries)
    with threadpool_limits(limits=args.threads, user_api="blas"):
        pools = threadpool_info()
        if not any(p.get("user_api") == "blas" for p in pools) or any(p["num_threads"] != args.threads for p in pools if p.get("user_api") == "blas"):
            raise ValueError("Could not verify the requested loaded BLAS thread count")
        setup_timing = timed(lambda: QueryTransform(model), args.warmups, args.repeats)
        transform = QueryTransform(model)
        timings = [benchmark_direction(d, images, texts, transform, q, args.warmups, args.repeats) for d, q in (("i2t", iquery), ("t2i", tquery))]
        comparison = [compare_direction(d, images, texts, model, transform, q, args.gallery_block) for d, q in (("i2t", iquery), ("t2i", tquery))]
    if (gallery_before != {"images": array_hash(images), "texts": array_hash(texts)}
            or model_before != {k: array_hash(getattr(model, k)) for k in model_before}):
        raise ValueError("A gallery or model array changed during diagnosis")
    verify_unchanged(watched)
    return {"study": "sanw_practical_v10_query_transform_diagnosis", "encoder": identity["encoder"], "family": identity["family"],
            "seed": identity["config"]["seed"], "selected_epoch": completion["selected_epoch"], "protocol_sha256": args.protocol_sha256,
            "scope": "completed selected state, training-only fixed query subsets against complete training source galleries; no accuracy or practical-goal claim",
            "identities_before_and_after": watched, "sources_inputs_checkpoints_and_arrays_unchanged": True,
            "heldout_inputs_opened": False, "original_mixed_split_archive_opened": False,
            "canonical_evaluation_unchanged": True, "checkpoint_reselection_or_update": False,
            "normalization": "exactly one Torch float32 F.normalize on raw gallery rows, then float64; transformed queries never normalized",
            "algebra": {"B": "U A V^T", "setup": "UA, V^T mu_t, (UA)^T mu_v",
                        "image_query": "qI = v + ((v-mu_v)^T UA) V^T; cI = -((v-mu_v)^T UA)(V^T mu_t)",
                        "text_query": "qT = t + ((t-mu_t)^T V)(UA)^T; cT = -((t-mu_t)^T V)((UA)^T mu_v)",
                        "complexity": "O(D r^2) setup, O(D r) per-query transform, unchanged D-dimensional gallery; no dense D-by-D B",
                        "score_vs_ranking": "Offsets restore the universal score, including comparisons across queries. A constant per-query offset is unnecessary for exact-arithmetic within-query ranking; reassociated float64 ties may differ."},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                            "blas_threads": args.threads, "torch_normalization_threads": 1, "loaded_thread_pools": pools},
            "measurement": {"warmups_per_case": args.warmups, "repeats_per_case": args.repeats, "gallery_comparison_block": args.gallery_block,
                            "timing_setup_only": setup_timing, "load_and_identity_checks_seconds": loading_seconds,
                            "not_end_to_end_retrieval_latency": True, "co_scheduled_jobs_must_be_absent": True},
            "queries": {"selection": "evenly spaced ascending training gallery positions; integer floor(i*(N-1)/(K-1)), first row if K=1",
                        "i2t_image_manifest_indices": iquery.tolist(), "i2t_image_ids": [image_ids[i] for i in iquery],
                        "t2i_source_gallery_indices": tquery.tolist(), "t2i_text_manifest_indices": source_rows[tquery].tolist(),
                        "t2i_text_ids": [text_ids[i] for i in tquery]},
            "unchanged_gallery_array_sha256": gallery_before, "timing_results": timings, "arithmetic_results": comparison,
            "diagnostic_total_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--gallery-block", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    for key in ("repository", "protocol", "completion", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an earlier diagnosis")
    result = diagnose(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "measured_training_only": True}))


if __name__ == "__main__":
    main()
