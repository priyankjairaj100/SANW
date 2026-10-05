"""Synthetic-only checks; no benchmark examples or outcomes are loaded."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_practical_prefix_reuse_v7 import verified_prior_chunks, independent_encode_reusing_prefix
from audit_practical_text_inference_v6 import independent_encode
from gcr.practical_prefix_capture_v7 import CapturingTower
from gcr.practical_text_inference_v6 import encode_token_rows
from gcr.practical_text_training_v6 import LastTextBlock
from gcr.practical_text_v6 import load_text_tower


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": digest(path)}


def tokens():
    rows = torch.zeros((4, 77), dtype=torch.int64)
    for index, content in enumerate(([321], [400, 14], [321], [500, 600, 700])):
        rows[index, :len(content) + 2] = torch.tensor([49406, *content, 49407])
    return rows


@pytest.mark.parametrize("encoder", ["vit_b32", "rn50"])
def test_prefix_reuse_matches_full_functional_tower_and_falls_back(tmp_path, encoder):
    torch.set_num_threads(1)
    tower = load_text_tower(encoder, ROOT / "data/practical_v6_models")
    initial, learned = LastTextBlock(tower).eval(), LastTextBlock(tower).eval()
    with torch.no_grad():
        learned.final_norm.bias.add_(torch.linspace(-.002, .002, 512))
    rows = tokens()
    captured = CapturingTower(tower, rows, tmp_path / encoder)
    production = encode_token_rows(captured, [initial, learned], rows)
    cache = captured.finish(production["canonical_prefix_sha256"])
    states = [initial.state_dict(), learned.state_dict()]
    full = independent_encode(encoder, ROOT / "data/practical_v6_models", states, rows)
    chunk = {"partition": 0, "unique_start": 0, "unique_stop": 3,
             "tokens_sha256": cache["tokens_sha256"], "canonical_prefix_sha256": full["canonical_prefix_sha256"],
             "unique_token_sequences": 3}
    evidence = {"schema": "verified_full_v6_audit_prefix_chunk_v7", "encoder": encoder,
                "chunk": chunk, "audit": {"synthetic_test_fixture": True}}
    reused = independent_encode_reusing_prefix(encoder, ROOT / "data/practical_v6_models", states, rows, cache, evidence)
    assert reused["prefix_audit_reuse"]["reused"]
    for field in ("delta", "learned", "reference"):
        assert np.array_equal(production[field], full[field])
        assert np.array_equal(reused[field], full[field])
        assert np.array_equal(reused[field][:, 0], reused[field][:, 2])
    assert np.count_nonzero(reused["delta"][0]) == 0
    assert np.count_nonzero(reused["delta"][1]) > 0
    # Even a self-consistently rehashed modified cache cannot evade the prior
    # independently audited ordered prefix digest.
    filename = Path(cache["directory"]) / "prefix_values.npy"
    values = np.load(filename, mmap_mode="r+")
    values[0, 0] += .01
    values.flush(); del values
    receipt_path = Path(cache["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["prefix_values.npy"] = digest(filename)
    modified_record = write(receipt_path, receipt)
    fallback = independent_encode_reusing_prefix(encoder, ROOT / "data/practical_v6_models", states, rows, modified_record, evidence)
    assert not fallback["prefix_audit_reuse"]["reused"]
    assert "ordered prefix digest" in fallback["prefix_audit_reuse"]["fallback_reason"]
    assert np.array_equal(fallback["delta"], full["delta"])


def synthetic_chain(tmp_path):
    rows = tokens().numpy()
    unique = rows[[0, 1, 3]]
    chunks = [{"partition": 0, "unique_start": 0, "unique_stop": 3,
               "tokens_sha256": hashlib.sha256(unique.tobytes()).hexdigest(),
               "canonical_prefix_sha256": "synthetic_prefix_digest", "unique_token_sequences": 3}]
    sources = {}
    for name in ("src/gcr/practical_text_v6.py", "src/gcr/practical_text_inference_v6.py", "scripts/audit_practical_text_inference_v6.py"):
        filename = tmp_path / name
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text("synthetic source identity")
        sources[name] = digest(filename)
    config = write(tmp_path / "config.json", {"synthetic": True})
    receipt = {"encoder": "vit_b32", "dataset_config": config["path"], "dataset_config_sha256": config["sha256"],
               "weight_identity": {"synthetic": True}, "input_hashes": {"synthetic": {"manifest": "unchanged"}},
               "source_hashes": sources, "environment": {"torch": str(torch.__version__)}}
    receipt_record = write(tmp_path / "receipt.json", receipt)
    meta = {"tokens_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "dataset": "synthetic",
            "prescore_receipt_sha256": receipt_record["sha256"], "canonical_prefix_chunks": chunks,
            "canonical_prefix_sha256": hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    runs = [{}]
    for number in range(3):
        metadata = write(tmp_path / f"metadata_{number}.json", meta)
        runs.append({"datasets": {"synthetic": {"text_encoding": {"metadata": metadata["path"], "metadata_sha256": metadata["sha256"]}}}})
    index = write(tmp_path / "index.json", {"encoder": "vit_b32", "status": "complete", "runs": runs,
                                            "prescore_receipt": receipt_record["path"], "prescore_receipt_sha256": receipt_record["sha256"]})
    analysis = write(tmp_path / "analysis.json", {"indices": {"vit_b32": index}})
    audit = write(tmp_path / "audit.json", {"passed": True,
                                          "source_sha256": "ebfaf2702f5d9a6f8a452097d07c9c729e2dfb1b107c8fc1963536aef0d25dcf",
                                          "analysis": analysis["path"], "analysis_sha256": analysis["sha256"]})
    return audit, receipt, rows, chunks


def test_full_audit_hash_chain_and_input_binding(tmp_path):
    audit, receipt, rows, chunks = synthetic_chain(tmp_path)
    result = verified_prior_chunks(tmp_path, audit, "vit_b32", "synthetic", receipt, rows, chunks)
    assert result[0]["audit"] == audit
    changed = dict(receipt, weight_identity={"synthetic": False})
    with pytest.raises(ValueError, match="weight_identity"):
        verified_prior_chunks(tmp_path, audit, "vit_b32", "synthetic", changed, rows, chunks)
    changed_rows = rows.copy(); changed_rows[0, 1] += 1
    with pytest.raises(ValueError, match="caption sequence"):
        verified_prior_chunks(tmp_path, audit, "vit_b32", "synthetic", receipt, changed_rows, chunks)
    with pytest.raises(ValueError, match="partitions/prefix"):
        verified_prior_chunks(tmp_path, audit, "vit_b32", "synthetic", receipt, rows, [dict(chunks[0], canonical_prefix_sha256="changed")])
    filename = Path(audit["path"])
    record = json.loads(filename.read_text()); record["passed"] = False
    failed = write(filename, record)
    with pytest.raises(ValueError, match="completed, passed"):
        verified_prior_chunks(tmp_path, failed, "vit_b32", "synthetic", receipt, rows, chunks)
