"""Optional reuse of prefix bytes already covered by a completed V6 audit.

No production module is imported. The trust chain is: externally pinned passed
audit -> its exact analysis/index/encoding metadata -> exact ordered prefix
digest. The new cache's token order and packed float32 bytes are rehashed before
any reuse. Every new learned/reference suffix is reconstructed functionally.
Any absent or mismatched reuse evidence falls back to the full independent tower.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_practical_text_inference_v6 import (
    _read_pinned_weights, _validated_suffixes, _suffix, independent_encode,
)


POLICY = "first_seen_full_token_row_then_batch1_own_eot_float32_prefix_bytes"
CANONICAL_SOURCES = (
    "src/gcr/practical_text_v6.py",
    "src/gcr/practical_text_inference_v6.py",
    "scripts/audit_practical_text_inference_v6.py",
)


def _read_verified(root, record):
    filename = Path(record["path"])
    filename = filename if filename.is_absolute() else Path(root) / filename
    with filename.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError(f"Evidence hash differs: {filename}")
    return json.loads(filename.read_text())


def verified_prior_chunks(root, audit_record, encoder, dataset, current_receipt, tokens, chunks):
    """Resolve chunk evidence from a hash-pinned *passed* full V6 audit.

    Returns partition-indexed evidence. Callers must catch validation failures
    and use the original full ``independently_encode`` routine instead. Only
    metadata and hashes are consulted here; no old test predictions are read.
    """
    audit = _read_verified(root, audit_record)
    if audit.get("passed") is not True or audit.get("source_sha256") != "ebfaf2702f5d9a6f8a452097d07c9c729e2dfb1b107c8fc1963536aef0d25dcf":
        raise ValueError("Reuse requires the completed, passed full V6 token audit")
    analysis_record = {"path": audit["analysis"], "sha256": audit["analysis_sha256"]}
    analysis = _read_verified(root, analysis_record)
    entry = analysis["indices"][encoder]
    index = _read_verified(root, entry)
    if index["status"] != "complete" or index["encoder"] != encoder or len(index["runs"]) != 4:
        raise ValueError("Prior audited encoder index is incomplete")
    receipt_record = {"path": index["prescore_receipt"], "sha256": index["prescore_receipt_sha256"]}
    old = _read_verified(root, receipt_record)
    for field in ("encoder", "dataset_config", "dataset_config_sha256", "weight_identity"):
        if old[field] != current_receipt[field]:
            raise ValueError(f"Prior prefix input identity differs: {field}")
    if old["input_hashes"][dataset] != current_receipt["input_hashes"][dataset]:
        raise ValueError("Prior benchmark input hashes differ")
    _read_verified(root, {"path": old["dataset_config"], "sha256": old["dataset_config_sha256"]})
    for source in CANONICAL_SOURCES:
        expected = old["source_hashes"][source]
        if current_receipt["source_hashes"].get(source) != expected:
            raise ValueError("Canonical prefix/audit source identity differs")
        filename = Path(root) / source
        with filename.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Canonical prefix/audit source bytes changed")
    if old["environment"]["torch"] != current_receipt["environment"]["torch"]:
        raise ValueError("Canonical Torch runtime differs")
    token_hash = hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()
    old_chunks = None
    metadata_records = []
    for run in index["runs"][1:]:
        entry = run["datasets"][dataset]["text_encoding"]
        record = {"path": entry["metadata"], "sha256": entry["metadata_sha256"]}
        meta = _read_verified(root, record)
        if meta["tokens_sha256"] != token_hash or meta["dataset"] != dataset or meta["prescore_receipt_sha256"] != receipt_record["sha256"]:
            raise ValueError("Audited caption sequence identity differs")
        if old_chunks is not None and meta["canonical_prefix_chunks"] != old_chunks:
            raise ValueError("Audited suffixes used different prefixes")
        old_chunks = meta["canonical_prefix_chunks"]
        expected_summary = hashlib.sha256(json.dumps(old_chunks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if meta["canonical_prefix_sha256"] != expected_summary:
            raise ValueError("Audited prefix summary differs")
        metadata_records.append(record)
    if chunks != old_chunks:
        raise ValueError("New canonical token partitions/prefix digests differ from audited V6")
    seen, unique = set(), []
    for row in np.asarray(tokens, dtype=np.int64):
        if row.tobytes() not in seen:
            seen.add(row.tobytes()); unique.append(row)
    unique = np.asarray(unique, dtype=np.int64)
    coverage, result = [], {}
    for number, chunk in enumerate(chunks):
        first, last = chunk["unique_start"], chunk["unique_stop"]
        if chunk["partition"] != number or not 0 <= first < last <= len(unique):
            raise ValueError("Invalid audited partition")
        if hashlib.sha256(unique[first:last].tobytes()).hexdigest() != chunk["tokens_sha256"]:
            raise ValueError("Reconstructed unique token partition differs")
        coverage.extend(range(first, last))
        result[number] = {"schema": "verified_full_v6_audit_prefix_chunk_v7", "encoder": encoder,
                          "dataset": dataset, "chunk": chunk, "audit": audit_record,
                          "analysis": analysis_record, "index": analysis["indices"][encoder],
                          "prescore_receipt": receipt_record, "encoding_metadata": metadata_records}
    if coverage != list(range(len(unique))):
        raise ValueError("Audited partition coverage differs")
    return result


def _verified_cache(cache_record, evidence, tokens):
    if evidence["schema"] != "verified_full_v6_audit_prefix_chunk_v7":
        raise ValueError("Unverified prior audit evidence")
    receipt = _read_verified(Path("/"), cache_record)
    if receipt["schema"] != "canonical_packed_prefix_capture_v7" or receipt["policy"] != POLICY:
        raise ValueError("Unsupported packed prefix policy")
    directory = Path(receipt["directory"])
    arrays = {}
    for name in ("prefix_values.npy", "prefix_offsets.npy", "prefix_tokens.npy"):
        filename = directory / name
        with filename.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["files"][name]:
                raise ValueError("Captured prefix file changed")
        arrays[name] = np.load(filename, mmap_mode="r", allow_pickle=False)
    values, offsets, stored_tokens = (arrays[name] for name in ("prefix_values.npy", "prefix_offsets.npy", "prefix_tokens.npy"))
    rows = np.asarray(tokens, dtype=np.int64)
    seen, unique, inverse = {}, [], []
    for row in rows:
        key = row.tobytes()
        if key not in seen:
            seen[key] = len(unique); unique.append(row)
        inverse.append(seen[key])
    unique = np.asarray(unique, dtype=np.int64)
    if stored_tokens.dtype != np.int64 or not np.array_equal(stored_tokens, unique):
        raise ValueError("Cache token order differs")
    if offsets.dtype != np.int64 or offsets.shape != (len(unique) + 1,) or offsets[0] != 0:
        raise ValueError("Invalid packed prefix offsets")
    expected_lengths = (unique == 49407).argmax(axis=1) + 1
    if not np.array_equal(np.diff(offsets), expected_lengths):
        raise ValueError("Packed prefixes differ from own-EOT lengths")
    if values.dtype != np.float32 or values.shape != (int(offsets[-1]), 512) or list(values.shape) != receipt["packed_shape"]:
        raise ValueError("Invalid packed prefix shape/dtype")
    chunk = evidence["chunk"]
    token_sha = hashlib.sha256(unique.tobytes()).hexdigest()
    if token_sha != receipt["tokens_sha256"] or token_sha != chunk["tokens_sha256"]:
        raise ValueError("Cache token digest differs from audited partition")
    if len(unique) != receipt["unique_token_sequences"] or len(unique) != chunk["unique_token_sequences"]:
        raise ValueError("Cache unique-token count differs")
    hasher = hashlib.sha256()
    for index, row in enumerate(unique):
        prefix = values[offsets[index]:offsets[index + 1]]
        if not np.isfinite(prefix).all():
            raise ValueError("Nonfinite cached prefix")
        hasher.update(row.tobytes()); hasher.update(prefix.tobytes())
    if hasher.hexdigest() != chunk["canonical_prefix_sha256"] or hasher.hexdigest() != receipt["canonical_prefix_sha256"]:
        raise ValueError("Reconstructed ordered prefix digest differs from full independent V6 audit")
    return values, offsets, unique, inverse, receipt


def independent_encode_reusing_prefix(encoder, model_directory, checkpoint_states, tokens, cache_record, evidence):
    """Reconstruct every new suffix, or fall back to full independent encoding.

    ``cache_record`` is an absolute receipt path plus sha256. ``evidence`` is one
    partition returned by ``verified_prior_chunks``. Output matches the original
    independent encoder and adds ``prefix_audit_reuse`` provenance.
    """
    torch.set_num_threads(1)
    rows = torch.as_tensor(tokens)
    if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 2 or rows.shape[1] != 77 or not len(rows):
        raise ValueError("Expected nonempty CPU int64 token rows")
    if torch.any(rows < 0) or torch.any(rows >= 49408) or not torch.all(rows[:, 0] == 49406) or not torch.all((rows == 49407).sum(dim=1) == 1):
        raise ValueError("Invalid CLIP token sequence")
    try:
        if evidence["encoder"] != encoder:
            raise ValueError("Reused prefix encoder differs")
        values, offsets, unique, inverse, receipt = _verified_cache(cache_record, evidence, rows.numpy())
    except (KeyError, TypeError, ValueError, OSError) as error:
        output = independent_encode(encoder, model_directory, checkpoint_states, rows)
        output["prefix_audit_reuse"] = {"reused": False, "fallback_reason": str(error)}
        return output
    pretrained, dimension, quick = _read_pinned_weights(encoder, model_directory)
    states = _validated_suffixes(checkpoint_states, pretrained)
    output = {name: np.empty((len(states), len(unique), dimension), dtype=np.float32)
              for name in ("delta", "learned", "reference")}
    with torch.inference_mode():
        for index in range(len(unique)):
            # A single-caption copy avoids writable-memmap aliasing and keeps
            # resident memory independent of the complete packed cache size.
            prefix = torch.from_numpy(np.array(values[offsets[index]:offsets[index + 1]], copy=True))[None]
            end = prefix.shape[1] - 1
            mask = torch.full((end + 1, end + 1), float("-inf"), dtype=torch.float32).triu_(1)
            old = _suffix(prefix, end, mask, states[0], quick, reference=True)
            for state_index, state in enumerate(states):
                new = _suffix(prefix, end, mask, state, quick, reference=False)
                if new.shape != (1, dimension) or not torch.isfinite(new).all() or not torch.isfinite(old).all():
                    raise ValueError("Invalid independent suffix output")
                output["learned"][state_index, index] = new[0].numpy()
                output["reference"][state_index, index] = old[0].numpy()
                output["delta"][state_index, index] = (new - old)[0].numpy()
    output = {name: value[:, inverse] for name, value in output.items()}
    output.update(unique_token_sequences=len(unique), canonical_prefix_sha256=receipt["canonical_prefix_sha256"],
                  prefix_audit_reuse={"reused": True, "cache": cache_record, "prior_full_audit": evidence["audit"],
                                      "chunk": evidence["chunk"],
                                      "scope": "prior independently verified prefix bytes; every current suffix reconstructed independently"})
    return output
