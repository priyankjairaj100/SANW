"""Disk-backed observation of unchanged canonical prefix calls.

This wrapper never substitutes cached values during production encoding. It
copies each returned prefix into a packed memmap for a later independent audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


POLICY = "first_seen_full_token_row_then_batch1_own_eot_float32_prefix_bytes"


def _digest(filename):
    with Path(filename).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CapturingTower:
    """Delegate a tower while capturing prefix bytes in first-seen token order.

    ``finish(encoded_prefix_sha256)`` must be called after ``encode_token_rows``.
    The returned receipt is also saved as ``prefix_cache_receipt.json``. Input
    tokens may contain duplicates; only their first occurrence is stored.
    """

    def __init__(self, tower, tokens, output_dir):
        self.tower = tower
        rows = np.asarray(tokens, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != 77 or not len(rows):
            raise ValueError("Expected nonempty caption token rows")
        if not np.all(rows[:, 0] == 49406) or not np.all((rows == 49407).sum(axis=1) == 1):
            raise ValueError("Expected leading SOT and exactly one EOT")
        if (rows < 0).any() or (rows >= 49408).any():
            raise ValueError("Token outside pinned vocabulary")
        seen, unique = set(), []
        for row in rows:
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                unique.append(row)
        self.tokens = np.stack(unique)
        lengths = (self.tokens == 49407).argmax(axis=1) + 1
        self.offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        filenames = ("prefix_values.npy", "prefix_offsets.npy", "prefix_tokens.npy", "prefix_cache_receipt.json")
        if any((self.output / name).exists() for name in filenames):
            raise FileExistsError("Never overwrite a captured prefix cache")
        self.values = np.lib.format.open_memmap(self.output / "prefix_values.npy", mode="w+",
                                               dtype=np.float32, shape=(int(self.offsets[-1]), 512))
        np.save(self.output / "prefix_offsets.npy", self.offsets, allow_pickle=False)
        np.save(self.output / "prefix_tokens.npy", self.tokens, allow_pickle=False)
        self.position, self.hasher, self.finished = 0, hashlib.sha256(), False

    def __getattr__(self, name):
        return getattr(self.tower, name)

    def eval(self):
        self.tower.eval()
        return self

    def prefix(self, tokens, trim=True):
        if self.finished or not trim or self.position >= len(self.tokens):
            raise ValueError("Unexpected canonical prefix call")
        tensor = torch.as_tensor(tokens)
        if tensor.device.type != "cpu" or tensor.dtype != torch.int64 or tensor.shape != (1, 77):
            raise ValueError("Prefix capture requires full CPU int64 batch-one tokens")
        if not np.array_equal(tensor.numpy()[0], self.tokens[self.position]):
            raise ValueError("Prefix call order differs from first-seen token order")
        prefix, ends = self.tower.prefix(tokens, trim=trim)
        first, last = self.offsets[self.position:self.position + 2]
        if prefix.device.type != "cpu" or prefix.dtype != torch.float32 or prefix.shape != (1, int(last - first), 512):
            raise ValueError("Noncanonical prefix tensor")
        if int(ends[0]) + 1 != last - first or not torch.isfinite(prefix).all():
            raise ValueError("Invalid own-EOT prefix")
        values = prefix.detach().numpy()
        self.values[first:last] = values[0]
        self.hasher.update(self.tokens[self.position].tobytes())
        self.hasher.update(values.tobytes())
        self.position += 1
        return prefix, ends

    def finish(self, encoded_prefix_sha256):
        if self.finished or self.position != len(self.tokens):
            raise ValueError("Prefix capture is incomplete or already finalized")
        if self.hasher.hexdigest() != encoded_prefix_sha256:
            raise ValueError("Captured prefixes differ from canonical production hash")
        self.values.flush()
        del self.values
        record = {"schema": "canonical_packed_prefix_capture_v7", "directory": str(self.output),
                  "policy": POLICY, "canonical_prefix_sha256": encoded_prefix_sha256,
                  "tokens_sha256": hashlib.sha256(self.tokens.tobytes()).hexdigest(),
                  "unique_token_sequences": len(self.tokens), "packed_shape": [int(self.offsets[-1]), 512],
                  "files": {name: _digest(self.output / name) for name in
                            ("prefix_values.npy", "prefix_offsets.npy", "prefix_tokens.npy")}}
        receipt_path = self.output / "prefix_cache_receipt.json"
        receipt_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        self.finished = True
        return {"path": str(receipt_path), "sha256": _digest(receipt_path), **record}
