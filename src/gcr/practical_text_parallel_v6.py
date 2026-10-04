"""Process-level parallelism around the unchanged single-caption token scorer.

Global token deduplication precedes deterministic caption partitioning. Every
worker uses one Torch thread and the same canonical single-caption helper.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from .practical_text_v6 import digest, load_text_tower
from .practical_text_training_v6 import LastTextBlock
from .practical_text_inference_v6 import encode_token_rows


def _worker(job):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(job["output"])
    record_path = output / "receipt.json"
    identity = {key: value for key, value in job.items() if key not in ("tokens",)}
    identity["tokens_sha256"] = hashlib.sha256(job["tokens"].tobytes()).hexdigest()
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["identity"] != identity:
            raise ValueError("Saved worker identity changed")
        for name, sha in record["files"].items():
            if digest(output / name) != sha:
                raise ValueError("Saved worker output changed")
        return record
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete worker output requires explicit recovery")
    output.mkdir(parents=True, exist_ok=True)
    tower = load_text_tower(job["encoder"], job["model_directory"])
    models = []
    for entry in job["checkpoints"]:
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Worker checkpoint changed")
        payload = torch.load(entry["path"], map_location="cpu", weights_only=True)
        model = LastTextBlock(tower)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models.append(model)
    result = encode_token_rows(tower, models, job["tokens"])
    for name in ("delta", "learned", "reference"):
        np.save(output / f"{name}.npy", result[name], allow_pickle=False)
    record = {"identity": identity, "canonical_prefix_sha256": result["canonical_prefix_sha256"],
              "unique_token_sequences": result["unique_token_sequences"],
              "files": {f"{name}.npy": digest(output / f"{name}.npy") for name in ("delta", "learned", "reference")}}
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def encode_parallel(encoder, model_directory, checkpoint_paths, tokens, output, workers=3, progress=None):
    if workers not in range(1, 7):
        raise ValueError("Choose one to six independent one-thread workers")
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 77 or not len(tokens):
        raise ValueError("Expected caption token rows")
    seen, unique_indices, inverse = {}, [], []
    for index, row in enumerate(tokens):
        key = row.tobytes()
        if key not in seen:
            seen[key] = len(unique_indices)
            unique_indices.append(index)
        inverse.append(seen[key])
    unique_indices, inverse = np.asarray(unique_indices, dtype=np.int64), np.asarray(inverse, dtype=np.int64)
    partitions = np.array_split(np.arange(len(unique_indices)), min(workers, len(unique_indices)))
    checkpoints = [{"path": str(Path(p).resolve()), "sha256": digest(p)} for p in checkpoint_paths]
    jobs = []
    for number, partition in enumerate(partitions):
        jobs.append({"encoder": encoder, "model_directory": str(Path(model_directory).resolve()), "checkpoints": checkpoints,
                     "output": str((Path(output) / f"worker_{number:02d}").resolve()), "partition": number,
                     "unique_start": int(partition[0]), "unique_stop": int(partition[-1]) + 1,
                     "tokens": tokens[unique_indices[partition]], "torch_threads": 1})
    records = {}
    if workers == 1:
        records[0] = _worker(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_worker, job): job["partition"] for job in jobs}
            for future in as_completed(futures):
                number = futures[future]
                records[number] = future.result()
                if progress:
                    progress({"completed_workers": len(records), "workers": len(jobs)})
    result, chunks = {}, []
    for name in ("delta", "learned", "reference"):
        parts = [np.load(Path(jobs[i]["output"]) / f"{name}.npy", allow_pickle=False) for i in range(len(jobs))]
        unique_values = np.concatenate(parts, axis=1)
        result[name] = unique_values[:, inverse]
        del parts, unique_values
    for number, job in enumerate(jobs):
        record = records[number]
        chunks.append({"partition": number, "unique_start": job["unique_start"], "unique_stop": job["unique_stop"],
                       "tokens_sha256": record["identity"]["tokens_sha256"],
                       "canonical_prefix_sha256": record["canonical_prefix_sha256"],
                       "unique_token_sequences": record["unique_token_sequences"]})
    result["canonical_prefix_chunks"] = chunks
    result["canonical_prefix_sha256"] = hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["unique_token_sequences"] = len(unique_indices)
    result["prefix_hash_policy"] = "SHA256 of canonical JSON ordered chunk receipts; each chunk hashes token row then canonical prefix bytes"
    return result
