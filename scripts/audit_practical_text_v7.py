#!/usr/bin/env python3
"""Independent v7 token inference, calibrated scores, statistics and gates.

Uses independent functional text-tower reconstruction. Imports no production
text encoder, scorer, evaluator, or statistics implementation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
from collections import defaultdict
import numpy as np
import torch
from audit_practical_text_inference_v6 import independent_encode
from audit_practical_prefix_reuse_v7 import verified_prior_chunks, independent_encode_reusing_prefix
from audit_practical_v6 import Audit as StatisticalAudit, path, read, digest, check_hash
ROOT = Path(__file__).resolve().parents[1]
ENCODERS, SEEDS = ("vit_b32", "rn50"), (17, 29, 43)
DATASETS = ("e_vil_test1000", "coco_karpathy", "sugarcrepe_pp")
FAMILY, NBOOT, RNG_SEED = 80, 100000, 20261007
STUDY_SCHEMA = "sanw_practical_source_pair_token_v7"
MODEL_FAMILY = "bounded_token_source_pair_calibrated_v7"
SOURCE_FACTORY = "gcr.practical_source_pair_v7.SourcePairTrainingExamples"
EVALUATION_STATUS = "adaptive_exploratory_after_historical_and_v6_test_reuse"
ALPHA_GRID = (.1, .2, .35, .5, .75, 1.)


def independently_validate_study(state, manifest, lock):
    """Audit study ownership and canonical selection without production imports."""
    if manifest.get("study_schema") != STUDY_SCHEMA or manifest.get("family") != MODEL_FAMILY or manifest.get("evaluation_status") != EVALUATION_STATUS or manifest.get("historical_and_v6_test_exposure") is not True:
        raise ValueError("Manifest does not identify this adaptive v7 study")
    amplitude = state["alpha"]
    if amplitude not in ALPHA_GRID or state["epoch"] not in range(1, 5):
        raise ValueError("Epoch/amplitude is outside the predeclared selection grid")
    record = state["training_ledger"]
    check_hash(record["path"], record["sha256"])
    ledger = read(record["path"])
    identity = ledger["identity"]
    expected_hash = hashlib.sha256(json.dumps(identity, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    if ledger["ledger_sha256"] != expected_hash or identity.get("schema") != STUDY_SCHEMA or identity.get("source_factory") != SOURCE_FACTORY:
        raise ValueError("Independent audit rejects a foreign or altered fitting ledger")
    if identity["protocol"] != {"path": lock["protocol"], "sha256": lock["protocol_sha256"]}:
        raise ValueError("Fitting protocol and selection protocol differ")
    protocol = read(lock["protocol"])
    if protocol["family"] != MODEL_FAMILY or protocol["source_factory"] != SOURCE_FACTORY or protocol["ownership"]["checkpoint_study_schema"] != STUDY_SCHEMA:
        raise ValueError("Wrong source-pair protocol")
    if protocol["development"]["alpha_grid"] != list(ALPHA_GRID):
        raise ValueError("Amplitude grid differs")
    parent = protocol["parent_protocols"][0]
    check_hash(parent["path"], parent["sha256"])
    if protocol["practical_gate"] != read(parent["path"])["practical_gate"]:
        raise ValueError("Inherited practical gate differs")
    if identity["source_sha256"] != protocol["source_hashes"]:
        raise ValueError("Fitting sources differ from protocol")
    for filename, sha in identity["source_sha256"].items():
        check_hash(filename, sha)
    for key, expected in protocol["fit_config"].items():
        if identity["config"].get(key) != expected:
            raise ValueError("Fitting hyperparameters differ from protocol")
    checkpoint = torch.load(path(state["checkpoint"]), map_location="cpu", weights_only=True)
    if checkpoint.get("study_schema") != STUDY_SCHEMA or checkpoint.get("source_factory") != SOURCE_FACTORY or checkpoint.get("schema") != "sanw_practical_text_last_block_v1":
        raise ValueError("Independent audit rejects a checkpoint from a different study")
    if checkpoint["ledger_sha256"] != expected_hash or checkpoint["config"] != identity["config"] or checkpoint["weight_identity"] != identity["weight_identity"]:
        raise ValueError("Checkpoint and fitting ledger disagree")
    if checkpoint["config"]["epsilon"] != .01 or checkpoint["optimizer_steps"] != state["optimizer_steps"]:
        raise ValueError("Trained correction budget or update count differs")
    tensors = checkpoint["state_dict"]
    squared = 0.
    for key, value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite suffix tensor")
        reference = "reference_" + key if key.startswith("block.") else key.replace("final_norm.", "reference_norm.") if key.startswith("final_norm.") else None
        if reference is not None:
            squared += float(torch.sum((value - tensors[reference]).to(torch.float64) ** 2))
    if squared <= 0 or abs(math.sqrt(squared) - state["update_norm"]) > 1e-10:
        raise ValueError("Trained weights do not match the selected nonzero update")
    snapshot_record = state["canonical_development_snapshot"]
    check_hash(snapshot_record["path"], snapshot_record["sha256"])
    snapshot = read(snapshot_record["path"])
    if snapshot["schema"] != "canonical_source_pair_calibration_v7" or snapshot["encoder"] != manifest["encoder"] or snapshot["seed"] != state["seed"]:
        raise ValueError("Wrong canonical development snapshot")
    if snapshot["trajectory_complete"] is not True or snapshot["selection_is_finalizable"] is not True or snapshot["held_out_evaluation"] != "not_performed" or snapshot["current_test_outcomes_used_for_selection"] is not False:
        raise ValueError("Canonical selection was incomplete or used current tests")
    check_hash(snapshot["binding"]["path"], snapshot["binding"]["sha256"])
    binding = read(snapshot["binding"]["path"])
    if binding["ledger"] != record or binding["protocol"] != identity["protocol"]:
        raise ValueError("Canonical development binding differs from fitting identity")
    candidates = snapshot["candidates"]
    expected_grid = sorted((epoch, a) for epoch in range(1, 5) for a in ALPHA_GRID)
    if sorted((r["epoch"], r["alpha"]) for r in candidates) != expected_grid:
        raise ValueError("Canonical selection grid is incomplete")
    eligible = []
    for candidate in candidates:
        gain = candidate["composition"]["paired_joint_accuracy"] - candidate["frozen_composition"]["paired_joint_accuracy"]
        if gain <= 0 or candidate["update_norm"] <= 0 or candidate["optimizer_steps"] <= 0 or candidate["residual_rms"] <= 1e-10:
            continue
        acceptable = True
        for direction in ("i2t", "t2i"):
            summary = candidate["retention"][direction]
            if (summary["family_size"], summary["replicates"], summary["bootstrap_seed"]) != (FAMILY, NBOOT, RNG_SEED):
                raise ValueError("Canonical development uncertainty contract changed")
            acceptable = acceptable and summary["difference"] >= 0 and summary["ci_lower"] > -.01
        if acceptable:
            key = (-gain, -candidate["composition"]["mean_paired_joint_margin"], candidate["epoch"], candidate["alpha"])
            eligible.append((key, candidate))
    if not eligible:
        raise ValueError("No eligible canonical development candidate")
    winner = sorted(eligible, key=lambda pair: pair[0])[0][1]
    selected = snapshot["selection"]
    if (winner["epoch"], winner["alpha"]) != (state["epoch"], amplitude) or (selected["selected_epoch"], selected["selected_alpha"]) != (state["epoch"], amplitude):
        raise ValueError("Independent development selection disagrees")
    if winner["checkpoint"] != {"path": state["checkpoint"], "sha256": state["checkpoint_sha256"]}:
        raise ValueError("Canonical selected checkpoint differs")
    return checkpoint


def _encoding_worker(job):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    states = []
    for record in job["checkpoints"]:
        check_hash(record["path"], record["sha256"])
        states.append(torch.load(path(record["path"]), map_location="cpu", weights_only=True)["state_dict"])
    if job.get("reuse_evidence") is not None:
        output = independent_encode_reusing_prefix(job["encoder"], ROOT / "data/practical_v6_models", states, job["tokens"], job["prefix_cache"], job["reuse_evidence"])
    else:
        output = independent_encode(job["encoder"], ROOT / "data/practical_v6_models", states, job["tokens"])
        output["prefix_audit_reuse"] = {"reused": False, "fallback_reason": "No matching prior full audit was bound"}
    return job["partition"], output


def independently_encode(encoder, checkpoints, tokens, chunks, workers, prefix_caches=None, evidence=None):
    seen, indices, inverse = {}, [], []
    for k, row in enumerate(tokens):
        key = row.tobytes()
        if key not in seen:
            seen[key] = len(indices)
            indices.append(k)
        inverse.append(seen[key])
    unique_tokens = tokens[indices]
    jobs = []
    coverage = []
    for chunk in chunks:
        first, last = chunk["unique_start"], chunk["unique_stop"]
        part = unique_tokens[first:last]
        if hashlib.sha256(part.tobytes()).hexdigest() != chunk["tokens_sha256"]:
            raise ValueError("Canonical token partition differs")
        coverage.extend(range(first, last))
        number = chunk["partition"]
        cache_by_partition = {r["partition"]: r["receipt"] for r in prefix_caches or []}
        jobs.append({"encoder": encoder, "checkpoints": checkpoints, "tokens": part, "partition": number,
                     "prefix_cache": cache_by_partition.get(number),
                     "reuse_evidence": evidence.get(number) if evidence is not None and number in cache_by_partition else None})
    if coverage != list(range(len(unique_tokens))) or [j["partition"] for j in jobs] != list(range(len(jobs))):
        raise ValueError("Canonical partition coverage is incomplete")
    results = {}
    if workers == 1:
        for job in jobs:
            k, value = _encoding_worker(job)
            results[k] = value
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_encoding_worker, job) for job in jobs]
            for future in as_completed(futures):
                k, value = future.result()
                results[k] = value
                print(json.dumps({"event": "independent_token_chunk", "encoder": encoder, "completed": len(results), "total": len(jobs)}), flush=True)
    for chunk in chunks:
        result = results[chunk["partition"]]
        if result["canonical_prefix_sha256"] != chunk["canonical_prefix_sha256"] or result["unique_token_sequences"] != chunk["unique_token_sequences"]:
            raise ValueError("Independent canonical prefix differs")
    arrays = {name: np.concatenate([results[i][name] for i in range(len(jobs))], axis=1)[:, inverse]
              for name in ("delta", "learned", "reference")}
    arrays["prefix_audit_reuse"] = [results[i]["prefix_audit_reuse"] for i in range(len(jobs))]
    return arrays


class Audit(StatisticalAudit):
    def token_retrieval(self, raw, manifest, arrays, delta, epsilon, alpha, metrics):
        if metrics["epsilon"] != epsilon or metrics["alpha"] != alpha:
            raise ValueError("Saved scoring parameters differ")
        image, text = arrays["image_features"].astype(np.float64), arrays["text_features"].astype(np.float64)
        self.equal(raw["image_ids"], arrays["image_ids"], "prediction_identity")
        self.equal(raw["text_ids"], arrays["text_ids"], "prediction_identity")
        ilook, tlook = ({str(v): k for k, v in enumerate(arrays[name])} for name in ("image_ids", "text_ids"))
        owners = np.full(len(text), -1, dtype=np.int64)
        for pair in manifest["pairs"]:
            ti = tlook[str(pair["text_id"])]
            if pair["relation"] != "source" or owners[ti] != -1:
                raise ValueError("Invalid ownership")
            owners[ti] = ilook[str(pair["image_id"])]
        if (owners < 0).any() or not np.array_equal(np.bincount(owners), np.full(len(image), 5)):
            raise ValueError("Incomplete ownership")
        self.equal(raw["text_source_image_ids"], arrays["image_ids"][owners], "prediction_identity")
        # A different direction traversal independently verifies both full pools.
        for direction, queries, count in (("i2t", image, len(text)), ("t2i", text, len(image))):
            ids, best = np.empty(len(queries), dtype=np.int64), np.empty(len(queries), dtype=np.float64)
            for start in range(0, len(queries), 37):
                if direction == "i2t":
                    baseline = np.dot(queries[start:start + 37], text.T)
                    correction = np.dot(queries[start:start + 37], delta.astype(np.float64).T)
                else:
                    baseline = np.dot(queries[start:start + 37], image.T)
                    correction = np.dot(delta[start:start + 37].astype(np.float64), image.T)
                scores = baseline + alpha * (epsilon * np.tanh(correction / epsilon))
                if not np.isfinite(scores).all() or np.max(np.abs(scores - baseline)) > alpha * epsilon + 1e-15:
                    raise ValueError("Unbounded or nonfinite token scores")
                for row in range(len(scores)):
                    # Stable full sorting is independent of production argmax.
                    ordered = np.argsort(-scores[row], kind="stable")
                    ids[start + row], best[start + row] = ordered[0], scores[row, ordered[0]]
            self.equal(ids, raw[f"{direction}_top_indices"], "full_gallery_top1")
            self.equal(best, raw[f"{direction}_top_scores"], "full_gallery_scores", tolerance=2e-12)
            correct = owners[ids] == np.arange(len(image)) if direction == "i2t" else ids == owners
            self.equal(correct, raw[f"{direction}_correct"], "retrieval_correctness")
            self.equal(sum(correct) / len(correct), metrics[direction]["r1"], "metric_aggregates")
            self.max_score_error = max(self.max_score_error, float(np.max(np.abs(best - raw[f"{direction}_top_scores"]))))

    def token_triplets(self, raw, manifest, arrays, delta, epsilon, alpha, metrics):
        rows = manifest["triplets"]
        ilook, tlook = ({str(v): k for k, v in enumerate(arrays[name])} for name in ("image_ids", "text_ids"))
        for key, source in (("item_ids", "id"), ("image_ids", "image_id"), ("categories", "category")):
            self.equal(raw[key], np.asarray([str(r[source]) for r in rows]), "prediction_identity")
        image = arrays["image_features"][[ilook[str(r["image_id"])] for r in rows]].astype(float)
        scores = {}
        for field in ("positive1", "positive2", "negative"):
            ids = np.asarray([str(r[f"{field}_id"]) for r in rows])
            self.equal(ids, raw[f"{field}_ids"], "prediction_identity")
            positions = [tlook[t] for t in ids]
            text = arrays["text_features"][positions].astype(float)
            change = delta[positions].astype(float)
            score = np.einsum("ij,ij->i", image, text) + alpha * (epsilon * np.tanh(np.einsum("ij,ij->i", image, change) / epsilon))
            self.equal(score, raw[f"{field}_scores"], "triplet_scores", tolerance=2e-12)
            scores[field] = score
            self.max_score_error = max(self.max_score_error, float(np.max(np.abs(score - raw[f"{field}_scores"]))))
        first, second = scores["positive1"] > scores["negative"], scores["positive2"] > scores["negative"]
        for field, value in (("positive1_correct", first), ("positive2_correct", second), ("correct", first & second)):
            self.equal(value, raw[field], "triplet_correctness")
        for field, value in (("positive1_accuracy", first), ("positive2_accuracy", second), ("both_accuracy", first & second)):
            self.equal(sum(value) / len(value), metrics[field], "metric_aggregates")

    def run_token(self, analysis_path, workers, frozen_indices):
        analysis = read(analysis_path)
        if analysis["family"] != MODEL_FAMILY or set(analysis["indices"]) != set(ENCODERS) or len(analysis["effects"]) != 10:
            raise ValueError("Incomplete token analysis")
        if analysis.get("evaluation_status") != EVALUATION_STATUS or analysis.get("study_schema") != STUDY_SCHEMA or analysis.get("historical_and_v6_test_exposure") is not True:
            raise ValueError("Analysis must disclose adaptive historical and v6 test reuse")
        check_hash(analysis["bootstrap_samples"], analysis["bootstrap_samples_sha256"])
        with np.load(path(analysis["bootstrap_samples"]), allow_pickle=False) as z:
            bootstrap_samples = {k: z[k] for k in z.files}
        lower_bounds = {}
        pooled = {}
        for filename in frozen_indices:
            index = read(filename)
            if index["status"] != "complete" or index["encoder"] in pooled:
                raise ValueError("Complete unique v6 frozen comparator indices required")
            pooled[index["encoder"]] = {"path": str(path(filename).relative_to(ROOT)), "sha256": digest(filename), "index": index}
        if set(pooled) != set(ENCODERS):
            raise ValueError("Both v6 frozen comparators required")
        # Load the exact pinned tokenizer source independently of production.
        import importlib.metadata, importlib.util
        distribution = importlib.metadata.distribution("open_clip_torch")
        if distribution.version != "2.32.0":
            raise ValueError("Tokenizer package version differs")
        spec = importlib.util.spec_from_file_location("independent_sanw_tokenizer", distribution.locate_file("open_clip/tokenizer.py"))
        token_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(token_module)
        tokenize = token_module.SimpleTokenizer()
        for encoder, entry in analysis["indices"].items():
            check_hash(entry["path"], entry["sha256"])
            index = read(entry["path"])
            if index["status"] != "complete" or index["encoder"] != encoder or len(index["runs"]) != 4:
                raise ValueError("Incomplete token scoring")
            check_hash(index["prescore_receipt"], index["prescore_receipt_sha256"])
            receipt = read(index["prescore_receipt"])
            check_hash(receipt["selection_lock"], receipt["selection_lock_sha256"])
            lock = read(receipt["selection_lock"])
            if receipt["selection_lock_sha256"] != analysis["selection_lock_sha256"] or lock["family"] != analysis["family"]:
                raise ValueError("Token analysis selection differs")
            if lock["family_size"] != FAMILY or lock["bootstrap_replicates"] != NBOOT or lock["bootstrap_seed"] != RNG_SEED:
                raise ValueError("Inference contract weakened")
            check_hash(lock["protocol"], lock["protocol_sha256"])
            if lock.get("study_schema") != STUDY_SCHEMA or lock.get("historical_and_v6_test_exposure") is not True or lock["evaluation_status"] != EVALUATION_STATUS:
                raise ValueError("V7 identity or adaptive exploratory status differs")
            if set(lock["encoders"]) != set(ENCODERS):
                raise ValueError("Both v7 encoders were not locked")
            for record in lock["encoders"].values():
                check_hash(record["manifest"], record["manifest_sha256"])
                selected = read(record["manifest"])
                if sorted(s["seed"] for s in selected["states"]) != list(SEEDS) or len(selected["states"]) != 3:
                    raise ValueError("Incomplete v7 seed grid")
                for state in selected["states"]:
                    check_hash(state["checkpoint"], state["checkpoint_sha256"])
            check_hash(receipt["manifest"], receipt["manifest_sha256"])
            manifest = read(receipt["manifest"])
            if receipt["manifest_sha256"] != lock["encoders"][encoder]["manifest_sha256"] or manifest["protocol_sha256"] != lock["protocol_sha256"]:
                raise ValueError("Token selected manifest changed")
            if manifest["test_outcomes_used_for_selection"] or manifest["current_test_outcomes_used_for_selection"]:
                raise ValueError("Invalid current test use")
            if sorted(s["seed"] for s in manifest["states"]) != list(SEEDS) or len(manifest["states"]) != 3:
                raise ValueError("Incomplete selected token seeds")
            for filename, sha in {**lock["source_hashes"], **manifest["source_hashes"]}.items():
                check_hash(filename, sha)
            check_hash(manifest["training_metadata"]["path"], manifest["training_metadata"]["sha256"])
            training_metadata = read(manifest["training_metadata"]["path"])
            check_hash(receipt["dataset_config"], receipt["dataset_config_sha256"])
            config = read(receipt["dataset_config"])
            if config["encoder"] != encoder:
                raise ValueError("Dataset encoder differs")
            if [r["state"] for r in index["runs"]] != receipt["states"] or receipt["states"][1:] != manifest["states"]:
                raise ValueError("Scored states differ from lock")
            epsilon, alpha = [], []
            checkpoints = []
            for state in manifest["states"]:
                check_hash(state["checkpoint"], state["checkpoint_sha256"])
                checkpoint = independently_validate_study(state, manifest, lock)
                if checkpoint["config"]["seed"] != state["seed"] or checkpoint["epoch"] != state["epoch"] or checkpoint["optimizer_steps"] <= 0 or state["update_norm"] <= 0 or abs(checkpoint["update_norm"] - state["update_norm"]) > 1e-12:
                    raise ValueError("Token trained state mismatch")
                if checkpoint["config"]["encoder"] != encoder or checkpoint["weight_identity"] != receipt["weight_identity"]:
                    raise ValueError("Token encoder provenance mismatch")
                epsilon.append(float(checkpoint["config"]["epsilon"]))
                alpha.append(float(state["alpha"]))
                checkpoints.append({"path": state["checkpoint"], "sha256": state["checkpoint_sha256"]})
            raw_by_state = {}
            for name in DATASETS:
                ds_manifest, arrays = self.load_data(config["datasets"][name], receipt["input_hashes"][name], encoder)
                metadata = read(config["datasets"][name]["metadata"])
                for key in ("model_revision", "weights_sha256", "open_clip_version", "logit_scale"):
                    if metadata.get(key) != training_metadata[key]:
                        raise ValueError("Frozen feature provenance differs")
                tokens = tokenize([r["text"] for r in ds_manifest["texts"]]).numpy()
                token_sha = hashlib.sha256(tokens.tobytes()).hexdigest()
                encodings, encoding_metas = [], []
                for run in index["runs"][1:]:
                    artifact = run["datasets"][name]["text_encoding"]
                    check_hash(artifact["archive"], artifact["archive_sha256"])
                    check_hash(artifact["metadata"], artifact["metadata_sha256"])
                    emeta = read(artifact["metadata"])
                    if emeta["state"] != run["state"] or emeta["dataset"] != name or emeta["prescore_receipt_sha256"] != index["prescore_receipt_sha256"] or emeta["tokens_sha256"] != token_sha or emeta["archive_sha256"] != artifact["archive_sha256"]:
                        raise ValueError("Text encoding provenance differs")
                    with np.load(path(artifact["archive"]), allow_pickle=False) as z:
                        self.equal(z["text_ids"], arrays["text_ids"], "encoding_identity")
                        encodings.append(z["delta"])
                    encoding_metas.append(emeta)
                chunks = encoding_metas[0]["canonical_prefix_chunks"]
                if any(m["canonical_prefix_chunks"] != chunks for m in encoding_metas):
                    raise ValueError("Selected suffixes did not use the same canonical prefixes")
                prefix_summary = hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if any(m["canonical_prefix_sha256"] != prefix_summary for m in encoding_metas):
                    raise ValueError("Canonical prefix chunk summary differs")
                caches = encoding_metas[0].get("prefix_cache_chunks")
                if any(m.get("prefix_cache_chunks") != caches for m in encoding_metas):
                    raise ValueError("Selected suffixes did not bind the same captured prefixes")
                evidence = None
                if lock.get("prior_v6_independent_audit") is not None and caches:
                    try:
                        evidence = verified_prior_chunks(ROOT, lock["prior_v6_independent_audit"], encoder, name, receipt, tokens, chunks)
                    except (KeyError, TypeError, ValueError, OSError) as error:
                        print(json.dumps({"event": "prefix_reuse_fallback", "encoder": encoder, "dataset": name, "reason": str(error)}), flush=True)
                independent = independently_encode(encoder, checkpoints, tokens, chunks, workers, caches, evidence)
                for reuse in independent["prefix_audit_reuse"]:
                    self.counts["reused_independently_audited_prefix_chunks" if reuse["reused"] else "fully_recomputed_prefix_chunks"] += 1
                for k, emeta in enumerate(encoding_metas):
                    # Exact equality is warranted by the fixed one-caption recipe.
                    if not np.array_equal(encodings[k], independent["delta"][k]):
                        raise ValueError("Independent token deltas differ bitwise")
                    for field in ("learned", "reference"):
                        if hashlib.sha256(independent[field][k].tobytes()).hexdigest() != emeta[field + "_sha256"]:
                            raise ValueError("Independent normalized text encoding differs")
                    self.counts["recomputed_token_vectors"] += len(encodings[k])
                del independent
                for k, run in enumerate(index["runs"]):
                    artifact = run["datasets"][name]
                    check_hash(artifact["predictions"], artifact["predictions_sha256"])
                    check_hash(artifact["metadata"], artifact["metadata_sha256"])
                    meta = read(artifact["metadata"])
                    if meta["metrics"] != artifact["metrics"] or meta["predictions_sha256"] != artifact["predictions_sha256"] or meta["provenance"] != {"state": run["state"], "dataset": name, "prescore_receipt_sha256": index["prescore_receipt_sha256"]}:
                        raise ValueError("Prediction provenance differs")
                    with np.load(path(artifact["predictions"]), allow_pickle=False) as z:
                        raw = {field: z[field] for field in z.files}
                    delta = np.zeros_like(arrays["text_features"]) if k == 0 else encodings[k - 1]
                    eps = .01 if k == 0 else epsilon[k - 1]
                    amplitude = 0. if k == 0 else alpha[k - 1]
                    if name == "sugarcrepe_pp":
                        self.token_triplets(raw, ds_manifest, arrays, delta, eps, amplitude, artifact["metrics"])
                    else:
                        self.token_retrieval(raw, ds_manifest, arrays, delta, eps, amplitude, artifact["metrics"])
                    if k == 0:
                        frozen_pooled = next(r for r in pooled[encoder]["index"]["runs"] if r["state"]["state_id"] == "frozen")
                        comparator = frozen_pooled["datasets"][name]
                        check_hash(comparator["predictions"], comparator["predictions_sha256"])
                        with np.load(path(comparator["predictions"]), allow_pickle=False) as z:
                            if set(z.files) != set(raw):
                                raise ValueError("Frozen prediction fields differ from prior v6")
                            for field in raw:
                                if not np.array_equal(raw[field], z[field]):
                                    raise ValueError(f"Frozen v7/v6 comparator differs bitwise: {field}")
                                self.counts["v6_frozen_bitwise_parity"] += int(np.size(raw[field]))
                    raw_by_state[(run["state"]["state_id"], name)] = raw
                    self.counts["prediction_archives"] += 1
            self.counts["states"] += 4
            selected = sorted(manifest["states"], key=lambda s: s["seed"])
            for name in DATASETS:
                frozen = raw_by_state[("frozen", name)]
                current = [raw_by_state[(s["state_id"], name)] for s in selected]
                for metric in (("both_accuracy",) if name == "sugarcrepe_pp" else ("i2t.r1", "t2i.r1")):
                    field = "correct" if metric == "both_accuracy" else metric.split(".")[0] + "_correct"
                    cluster_key = "text_source_image_ids" if metric == "t2i.r1" else "image_ids"
                    delta = np.array([r[field].astype(float) - frozen[field].astype(float) for r in current])
                    samples = self.bootstrap(delta, frozen[cluster_key])
                    eid = f"{encoder}__{name}__{metric.replace('.', '_')}"
                    self.equal(samples, bootstrap_samples[eid], "bootstrap_values", tolerance=2e-12)
                    self.max_bootstrap_error = max(self.max_bootstrap_error, float(np.max(np.abs(samples - bootstrap_samples[eid]))))
                    ordered = sorted(samples.tolist())
                    def quantile(p):
                        at = p * (len(ordered) - 1)
                        lo, hi = math.floor(at), math.ceil(at)
                        return ordered[lo] + (at - lo) * (ordered[hi] - ordered[lo])
                    low, high = quantile(.05 / (2 * FAMILY)), quantile(1 - .05 / (2 * FAMILY))
                    effect = next(r for r in analysis["effects"] if r["effect_id"] == eid)
                    for key, value in {"difference": delta.mean(), "ci_lower": low, "ci_upper": high,
                                       "seed_differences": delta.mean(axis=1), "frozen": frozen[field].mean(),
                                       "selected": np.array([r[field] for r in current]).mean()}.items():
                        self.equal(value, effect[key], "effect_aggregates", tolerance=2e-12)
                    if effect["family_size"] != FAMILY or effect["replicates"] != NBOOT or effect["bootstrap_seed"] != RNG_SEED:
                        raise ValueError("Inference contract changed")
                    lower_bounds[(encoder, name, metric)] = low
                    self.counts["effects"] += 1
        gates = []
        for encoder in ENCODERS:
            retained = all(lower_bounds[(encoder, "e_vil_test1000", m)] > -.01 for m in ("i2t.r1", "t2i.r1"))
            improved = lower_bounds[(encoder, "sugarcrepe_pp", "both_accuracy")] > 0
            coco = all(lower_bounds[(encoder, "coco_karpathy", m)] > -.01 for m in ("i2t.r1", "t2i.r1"))
            gates.append({"encoder": encoder, "nonzero_three_seeds": True, "retrieval_retention": retained,
                          "caption_improvement": improved, "passed": retained and improved, "secondary_coco_retention": coco})
        if gates != analysis["gates"] or all(g["passed"] for g in gates) != analysis["cross_encoder_passed"]:
            raise ValueError("Practical gate verdict differs")
        if all(g["passed"] and g["secondary_coco_retention"] for g in gates) != analysis["cross_encoder_with_coco_passed"]:
            raise ValueError("Transfer gate verdict differs")
        return {"schema_version": 1, "passed": True, "analysis": str(path(analysis_path).relative_to(ROOT)),
                "analysis_sha256": digest(analysis_path), "source_sha256": digest(__file__), "counts": dict(self.counts),
                "max_score_difference": self.max_score_error, "max_bootstrap_difference": self.max_bootstrap_error,
                "independence": "Pinned prefixes fully reconstructed or byte-verified against a successful full v6 audit; every current learned/reference suffix, full-gallery score, SC++ pair and all 1M bootstrap values independently reconstructed",
                "prior_v6_independent_audit": lock.get("prior_v6_independent_audit"),
                "gates": gates, "v6_frozen_comparators": {e: {"path": v["path"], "sha256": v["sha256"]} for e, v in pooled.items()},
                "evaluation_status": EVALUATION_STATUS, "study_schema": STUDY_SCHEMA, "historical_and_v6_test_exposure": True,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--frozen-v6-indices", nargs=2, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = Audit().run_token(args.analysis, args.workers, args.frozen_v6_indices)
    output = path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
